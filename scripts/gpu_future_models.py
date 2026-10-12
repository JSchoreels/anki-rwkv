"""Isolated Python models, task specialists (F13), and capacity variants (F11).

These prototypes leave the source consumed by the approved F01–F08 run unchanged.
Python controls are required when benchmarking Python candidate rewrites.
"""

import ast
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F


@lru_cache(maxsize=2)
def python_model_class(model_name):
    package = {"baseline": "rwkv", "expertium": "expertium"}[model_name]

    def clone(module_name, overrides):
        original = importlib.import_module(module_name)
        source = inspect.getsource(original)
        tree = ast.parse(source)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        class OneLayerRatios(ast.NodeTransformer):
            def visit_BinOp(self, node):
                self.generic_visit(node)
                denominator = node.right
                if (isinstance(node.op, ast.Div) and isinstance(denominator, ast.BinOp)
                        and isinstance(denominator.op, ast.Sub) and isinstance(denominator.left, ast.Attribute)
                        and denominator.left.attr in ("n_layers", "total_layers")
                        and isinstance(denominator.right, ast.Constant) and denominator.right.value == 1):
                    node.right = ast.Call(ast.Name("max", ast.Load()), [denominator, ast.Constant(1)], [])
                return node
        # Baseline initialization divides by n_layers-1. Define its one-layer
        # endpoint in the isolated F11 class; existing multilayer ratios are exact.
        classes = [OneLayerRatios().visit(node) for node in classes]
        for cls in classes:
            for method in cls.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = [node for node in method.decorator_list
                                             if not isinstance(node, ast.Name) or node.id != "FunctionType"]
        name = "_gpu_future_" + module_name.replace(".", "_")
        module = types.ModuleType(name)
        module.__dict__.update(original.__dict__)
        module.__dict__.update(overrides)
        module.__dict__.update(__name__=name, ModuleType=torch.nn.Module)
        sys.modules[name] = module
        exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), original.__file__, "exec"), module.__dict__)
        return module

    recurrent = clone(package + ".model.rwkv_model", {})
    srs = clone(package + ".model.srs_model", {"RWKV7": recurrent.RWKV7})
    return srs.SrsRWKV


def capacity_variant(architecture, width_heads=None, remove_layers=0):
    """Keep all five streams and K=32; never mutate a shared default config."""
    if width_heads is not None and (not isinstance(width_heads, int) or width_heads < 1):
        raise ValueError("Width requires at least one 32-dimensional head")
    if not isinstance(remove_layers, int) or remove_layers < 0:
        raise ValueError("remove_layers must be a nonnegative integer")
    width = architecture.d_model if width_heads is None else 32 * width_heads
    modules = []
    for name, config in architecture.modules:
        layers = max(1, config.n_layers - remove_layers)
        heads = config.n_heads if width_heads is None else width_heads
        modules.append((name, replace(config, d_model=width, n_heads=heads, n_layers=layers,
                                      total_layers=layers, layer_offset=0)))
    return replace(architecture, d_model=width, modules=modules)


@dataclass
class TaskStatistics:
    average_loss: torch.Tensor
    curve_loss: torch.Tensor | None
    instant_rating_loss: torch.Tensor | None
    p_curve: torch.Tensor | None
    p_imm: torch.Tensor | None
    p_imm_all: torch.Tensor | None
    ahead_n: int
    imm_n: int
    has_label: torch.Tensor
    is_query: torch.Tensor
    label_rating: torch.Tensor
    label_review_th: torch.Tensor
    curve_logits: torch.Tensor | None = None


class TaskModel(torch.nn.Module):
    """Same backbone; remove unused heads and losses for Curve/Instant specialists."""

    def __init__(self, source, task="joint", curve_weight=1.0, instant_weight=1.0):
        super().__init__()
        if task not in ("joint", "curve", "instant"):
            raise ValueError("Task must be joint, curve or instant")
        if curve_weight < 0 or instant_weight < 0 or (task == "joint" and curve_weight + instant_weight == 0):
            raise ValueError("Task weights must be nonnegative and at least one must be positive")
        self.task, self.curve_weight, self.instant_weight = task, curve_weight, instant_weight
        inactive = ({"head_p", "p_linear"} if task == "curve" else
                    {"head_ahead_logits", "ahead_linear", "head_w", "w_linear"} if task == "instant" else set())
        for name, child in source.named_children():
            if name not in inactive:
                self.add_module(name, child)
        for name in ("d_model", "num_curves", "num_points", "s_point_spread", "s_max", "point_spread", "max_e"):
            setattr(self, name, getattr(source, name))
        # Unbound helpers do not retain the source model or its removed heads.
        self._curve = type(source).forgetting_curve
        self._interp = type(source).interp

    def forward_features(self, batch):
        x = self.features2card(batch.start)
        if len(batch.sub_gather) != len(self.rwkv_modules):
            raise ValueError("Prepared hierarchy does not match model streams")
        for i, stream in enumerate(self.rwkv_modules):
            outputs = []
            for gather, length, shift, skip in zip(batch.sub_gather[i], batch.sub_gather_lens[i],
                                                   batch.time_shift_selects[i], batch.skips[i]):
                inputs = torch.index_select(x, 0, gather.clamp(min=0)).view(-1, length, self.d_model)
                outputs.append(stream(inputs, shift.view(-1, length), skip.view(-1, length)).reshape(-1, self.d_model))
            x = torch.cat(outputs)
        return self.prehead_dropout(self.prehead_norm(x.view(batch.num_data, -1, self.d_model)))

    def get_loss(self, batch):
        x = self.forward_features(batch)
        elapsed, _, label_y, rating, has_label, _, query = batch.labels.float().unbind(-1)
        curve_mask, instant_mask = (1 - query) * has_label, query * has_label
        rating = (rating - 1).clamp(min=0).long()
        zero = x.sum() * 0
        curve_objective = instant_objective = zero
        curve_loss = instant_loss = p_curve = p_imm = p_all = corrected = None

        def average(value, mask):
            return (value * mask).sum() / (1e-8 + mask.sum())

        if self.task != "instant":
            logits_w = self.w_linear(self.head_w(x).float())
            log_w, w = logits_w.log_softmax(-1), logits_w.softmax(-1)
            logits_ahead = self.ahead_linear(self.head_ahead_logits(x).float())
            raw_curve = self._curve(self, w, elapsed.unsqueeze(-1))
            raw_logits = torch.log(raw_curve / (1 - raw_curve))
            corrected = raw_logits + self._interp(self, logits_ahead, elapsed.unsqueeze(-1))
            p_curve = corrected.sigmoid()
            curve_loss = average(F.binary_cross_entropy_with_logits(corrected, label_y, reduction="none"), curve_mask)
            raw_loss = average(F.binary_cross_entropy_with_logits(raw_logits, label_y, reduction="none"), curve_mask)
            w_loss = average(F.kl_div(log_w, torch.ones_like(w) / self.num_curves, reduction="none").mean(-1), curve_mask)
            magnitude = average((1e-16 + logits_ahead.square().mean(-1)).sqrt(), curve_mask)
            difference = average((1e-16 + logits_ahead.diff().square().mean(-1)).sqrt(), curve_mask)
            curve_objective = 0.5 * curve_loss + 0.5 * raw_loss + 1e-5 * w_loss + 1e-4 * magnitude + 1e-3 * difference
        if self.task != "curve":
            logits_p = self.p_linear(self.head_p(x).float())
            p_all = logits_p.softmax(-1)
            p_imm = (1 - p_all[..., 0]).clamp(1e-5, 1 - 1e-5)
            ratings_loss = F.cross_entropy(logits_p.flatten(0, 1), rating.flatten(), reduction="none").view_as(rating)
            instant_loss = average(ratings_loss, instant_mask)
            instant_objective = instant_loss
        objective = ((self.curve_weight * curve_objective + self.instant_weight * instant_objective) if self.task == "joint"
                     else curve_objective if self.task == "curve" else instant_objective)
        return TaskStatistics(objective, curve_loss, instant_loss, p_curve, p_imm, p_all,
                              int(curve_mask.sum()) if self.task != "instant" else 0,
                              int(instant_mask.sum()) if self.task != "curve" else 0,
                              has_label.detach(), query.detach(), rating.detach(), batch.label_review_th.detach(), corrected)


def load_pretrained(model, checkpoint, manifest_path, forbidden_users):
    """Require exact shapes and explicit prior data/exposure; preserve the source file."""
    checkpoint, manifest_path = Path(checkpoint), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    required = {"sha256", "dataset_revision", "training_users", "training_reviews", "validation_users", "provenance"}
    if not required.issubset(manifest):
        raise ValueError("Pretraining manifest is missing exposure/data/provenance fields")
    overlap = set(forbidden_users) & (set(manifest["training_users"]) | set(manifest["validation_users"]))
    if overlap:
        raise ValueError("Pretraining has seen reserved evaluation/final-test users")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint fingerprint does not match its manifest")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return {"checkpoint": str(checkpoint), "manifest": manifest, "optimizer": "fresh; prior review exposure must be included separately"}
