"""Opt-in experiments for preparation and CUDA Graph training; original paths remain available."""
from dataclasses import dataclass
import ast
import inspect
import math
import numpy as np
import torch
from rwkv.config import DAY_OFFSET_ENCODE_PERIODS, ID_ENCODE_DIMS, ID_SPLIT, RWKV_SUBMODULES
from rwkv.data_processing import RWKVSample
from rwkv.model.srs_model import PreparedBatch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.prepare_batch import greedy_splits


@dataclass
class DeferredPreparedBatch(PreparedBatch):
    encodings: list
    global_T: int

    def to(self, device):
        batch = super().to(device)
        columns = [batch.start]
        # Transfer four lookup tables and their indices, rather than full encodings.
        for index in range(4):
            tables, indices, offset = [], [], 0
            for parts, days, first, baselines in self.encodings:
                table, inverse = parts[index]
                tables.append(table)
                indices.append(torch.nn.functional.pad(inverse + offset, (0, self.global_T-len(inverse))))
                offset += len(table)
            table = torch.cat(tables).to(device)
            inverse = torch.cat(indices).to(device)
            columns.append(table[inverse])
        days = torch.cat([torch.nn.functional.pad(part[1], (0, self.global_T-len(part[1])))
                          for part in self.encodings]).to(device)
        first = torch.cat([torch.nn.functional.pad(part[2], (0, self.global_T-len(part[2])))
                           for part in self.encodings]).to(device)
        baselines = torch.stack([part[3] for part in self.encodings]).repeat_interleave(self.global_T, 0).to(device)
        periods = torch.tensor(DAY_OFFSET_ENCODE_PERIODS, device=device)
        frequencies = (2*np.pi/periods).float()
        angles = frequencies * ((baselines + days[:, None]) % periods)
        first_angles = frequencies * ((baselines + first[:, None]) % periods)
        columns.append(torch.stack((angles.sin(), angles.cos(), first_angles.sin(), first_angles.cos()), -1).flatten(1))
        batch.start = torch.cat(columns, -1)
        padded = torch.cat([torch.arange(self.global_T) >= len(part[1]) for part in self.encodings]).to(device)
        batch.start.masked_fill_(padded[:, None], 0)
        return batch


def prepare_fast(data_list: list[RWKVSample], target_len=None, seed=None, gpu_encodings=False,
                 batched_id_encodings=False, layout=None, padding_factor=0.9,
                 allowed_excess_in_one_step=20000, consistent_seed=None) -> PreparedBatch:
    # consistent_seed (SM21 state carry): id encodings and day baselines are functions of (user seed, stream, id),
    # identical in every window of the user (scripts.state_carry_train); None = the per-sample draws below.
    if seed is not None:
        if batched_id_encodings or layout is not None:
            # These preparation extensions operate on CPU workers only.
            torch.random.default_generator.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    with torch.no_grad():
        global_T = max([data.card_features.size(0) for data in data_list])
        data_list_t_sum = sum([data.card_features.size(0) for data in data_list])

        feature_parts = []

        def add_encodings(card_features, day_offsets, day_offsets_first, ids):
            def generate_id_encoding(submodule):
                ENCODE_DIM = ID_ENCODE_DIMS[submodule]
                return torch.randint(
                    low=0,
                    high=ID_SPLIT,
                    size=(ENCODE_DIM,),
                    device=card_features.device,
                    requires_grad=False,
                ).to(card_features.dtype) - ((ID_SPLIT - 1) / 2)

            gather = [card_features]
            id_parts = []
            for submodule in RWKV_SUBMODULES:
                if submodule == "user_id":
                    continue
                values = ids[submodule].numpy()
                # Preserve the original set order and per-ID RNG draws.
                if consistent_seed is not None:
                    from scripts.state_carry_train import consistent_id_table
                    table, indices = consistent_id_table(ids[submodule], ID_ENCODE_DIMS[submodule], card_features.dtype,
                                                         consistent_seed, submodule)
                elif batched_id_encodings:
                    from scripts.gpu_id_encodings import batched_id_table
                    table, indices = batched_id_table(ids[submodule], ID_ENCODE_DIMS[submodule], card_features.dtype)
                else:
                    encode = {identifier: generate_id_encoding(submodule)
                              for identifier in set(ids[submodule].tolist())}
                    unique, inverse = np.unique(values, return_inverse=True)
                    table = torch.stack([encode[identifier] for identifier in unique])
                    indices = torch.from_numpy(inverse)
                if gpu_encodings:
                    id_parts.append((table, indices))
                else:
                    gather.append(table[indices])
            baselines = []
            for period in DAY_OFFSET_ENCODE_PERIODS:
                if consistent_seed is not None:
                    from scripts.state_carry_train import consistent_baseline
                    baseline = consistent_baseline(consistent_seed, period)
                else:
                    baseline = torch.randint(low=0, high=period, size=(1,))
                baselines.append(baseline)
                if not gpu_encodings:
                    f = 2 * np.pi / period
                    angle = f * ((baseline + day_offsets) % period)
                    first_angle = f * ((baseline + day_offsets_first) % period)
                    gather.extend([torch.stack((angle.sin(), angle.cos()), -1).to(card_features.dtype),
                                   torch.stack((first_angle.sin(), first_angle.cos()), -1).to(card_features.dtype)])
            if gpu_encodings:
                feature_parts.append((id_parts, day_offsets, day_offsets_first, torch.cat(baselines)))
                return card_features
            return torch.cat(gather, dim=-1)

        card_features_with_ids = [
            add_encodings(
                data.card_features, data.day_offsets, data.day_offsets_first, data.ids
            )
            for data in data_list
        ]
        if layout is not None:
            if (layout.num_data != len(data_list) or layout.labels.shape[:2] != (len(data_list), global_T)
                    or layout.start.shape != (len(data_list) * global_T, data_list[0].card_features.shape[1])):
                raise ValueError("Cached layout does not match the samples")
            if gpu_encodings:
                return DeferredPreparedBatch(**layout.__dict__, encodings=feature_parts, global_T=global_T)
            from dataclasses import replace
            start = torch.cat([torch.nn.functional.pad(features, (0, 0, 0, global_T - features.shape[0]))
                               for features in card_features_with_ids])
            return replace(layout, start=start)
        start_tensor = torch.cat(
            [
                torch.nn.functional.pad(
                    card_features, (0, 0, 0, global_T - card_features.size(0))
                )
                for card_features in card_features_with_ids
            ],
            dim=0,
        )

        # Interpretation: the element representing a review_th of i is currently at a[i] where a[i] is a 1D tensor that holds all the data
        boundary_offset = 0
        current_locs_list = [
            i * global_T
            + torch.arange(0, data.card_features.size(0), 1, dtype=torch.long)
            for i, data in enumerate(data_list)
        ]

        # total used mem = x(1+f) where x is the sum of seq lens, f is the factor
        # at MAX and t, we use MAX*(1+t) memory
        # so f = MAX*(1+t)/x - 1
        factor = padding_factor
        if target_len is None:
            splits = greedy_splits(data_list, factor=factor,
                                   allowed_excess_in_one_step=allowed_excess_in_one_step)
        else:
            splits = greedy_splits(
                data_list, factor=target_len * (1 + factor) / data_list_t_sum - 1,
                allowed_excess_in_one_step=allowed_excess_in_one_step,
            )
        sub_gather = []
        sub_skip_gather = []
        sub_time_shift_gather = []
        sub_gather_lens = []
        for submodule_name, _ in DEFAULT_ANKI_RWKV_CONFIG.modules:
            assert submodule_name in splits
            split = splits[submodule_name]

            all_offset = 0
            next_locs_list = [
                np.zeros(data.card_features.size(0), dtype=np.int64)
                for data in data_list
            ]
            gather_lens = []
            gather = []
            skip_gather = []
            time_shift_gather = []
            for split_i in range(len(split)):
                l = 0 if split_i == 0 else split[split_i - 1]
                r = split[split_i]
                gather_lens.append(r)
                take_list = []
                skip_list = []
                time_shift_list = []

                for data_i, (data, current_locs) in enumerate(
                    zip(data_list, current_locs_list)
                ):
                    split_len = data.modules[submodule_name].split_len
                    split_B = data.modules[submodule_name].split_B
                    boundary_offset = 0
                    boundaries = []
                    for s_l, s_b in zip(split_len, split_B):
                        boundaries.append(boundary_offset)
                        boundary_offset += s_l * s_b

                    boundaries.append(boundary_offset)
                    assert boundary_offset == data.card_features.size(0)

                    module_data = data.modules[submodule_name]
                    for module_data_i, (data_split_B, data_split_len) in enumerate(
                        zip(module_data.split_B, module_data.split_len)
                    ):
                        if l < data_split_len and data_split_len <= r:
                            from_slice = module_data.from_perm[
                                boundaries[module_data_i] : boundaries[
                                    module_data_i + 1
                                ]
                            ]
                            take_from = torch.index_select(
                                current_locs, dim=0, index=from_slice
                            ).view(data_split_B, data_split_len)

                            # Maybe random instead of 0 padding to reduce collisions
                            take_from = torch.nn.functional.pad(
                                take_from,
                                (0, r - data_split_len),
                                mode="constant",
                                value=-1,
                            )
                            take_list.append(take_from)

                            skip = torch.index_select(
                                data.skips, dim=0, index=from_slice
                            ).view(data_split_B, data_split_len)
                            skip_arr = skip.numpy()
                            assert (skip_arr[0] == False).any()
                            positions = np.broadcast_to(np.arange(data_split_len), skip_arr.shape)
                            previous = np.maximum.accumulate(np.where(skip_arr, 0, positions), axis=1)
                            time_shift_select = np.zeros(skip_arr.shape, dtype=np.int32)
                            time_shift_select[:, 1:] = previous[:, :-1]

                            skip = torch.nn.functional.pad(
                                skip,
                                (0, r - data_split_len),
                                mode="constant",
                                value=True,
                            )
                            skip_list.append(skip)
                            time_shift_select = torch.nn.functional.pad(
                                torch.tensor(
                                    time_shift_select,
                                    dtype=torch.int32,
                                    device=skip.device,
                                ),
                                (0, r - data_split_len),
                                mode="constant",
                                value=0,
                            )
                            time_shift_list.append(time_shift_select)

                            destinations = (all_offset + np.arange(data_split_B)[:, None] * r
                                            + np.arange(data_split_len)[None, :])
                            next_locs_list[data_i][from_slice.numpy()] = destinations.reshape(-1)
                            all_offset += data_split_B * r
                gather.append(torch.cat(take_list, dim=0).flatten())
                skip_gather.append(torch.cat(skip_list, dim=0).flatten())
                time_shift_gather.append(
                    torch.cat(time_shift_list, dim=0).flatten().long()
                )

            sub_gather.append(gather)
            next_locs_list = [torch.tensor(x) for x in next_locs_list]
            current_locs_list = next_locs_list
            sub_gather_lens.append(gather_lens)
            sub_skip_gather.append(skip_gather)
            sub_time_shift_gather.append(time_shift_gather)

        def pad_labels(labels):
            return torch.nn.functional.pad(
                labels, (0, 0, 0, global_T - labels.size(0)), mode="constant", value=0
            )

        padded_labels = torch.stack(
            list(map(lambda data: pad_labels(data.global_labels), data_list))
        )

        def pad_review_ths(labels):
            return torch.nn.functional.pad(
                labels, (0, global_T - labels.size(0)), mode="constant", value=-1
            )

        padded_label_review_th = torch.stack(
            list(map(lambda data: pad_review_ths(data.label_review_ths), data_list))
        )
        batch = PreparedBatch(
            num_data=len(data_list),
            start=start_tensor,
            sub_gather=sub_gather,
            sub_gather_lens=sub_gather_lens,
            skips=sub_skip_gather,
            time_shift_selects=sub_time_shift_gather,
            labels=padded_labels,
            label_review_th=padded_label_review_th,
        )
        if gpu_encodings:
            return DeferredPreparedBatch(**batch.__dict__, encodings=feature_parts, global_T=global_T)
        return batch


@torch.no_grad()
def copy_downcast_fast(master, child):
    sources = dict(master.named_parameters())
    for name, parameter in child.named_parameters():
        # copy_ performs the cast in the copy kernel, avoiding a temporary.
        parameter.copy_(sources[name])


@torch.no_grad()
def transfer_grads_fast(master, child):
    targets = dict(master.named_parameters())
    for name, parameter in child.named_parameters():
        if parameter.grad is None:
            continue
        target = targets[name]
        if target.grad is None:
            target.grad = torch.empty_like(target)
            target.grad.copy_(parameter.grad)
        else:
            # Retain accumulation semantics for callers that already have gradients.
            target.grad.add_(parameter.grad)
        parameter.grad.zero_()


class _CaptureLoss(ast.NodeTransformer):
    def visit_If(self, node):
        # Validate finiteness immediately after replay instead of synchronizing
        # with the CPU inside capture. Loss and gradient checks remain enabled.
        if "isnan" in ast.unparse(node.test):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1
                and isinstance(node.args[0], ast.Call) and isinstance(node.args[0].func, ast.Attribute)
                and node.args[0].func.attr == "item"):
            return node.args[0].func.value
        return node


class GraphTrainingAdapter:
    """Capture forward, loss and backward together on one stream; optimizer stays outside.

    Supports one repeated shape. General histories need a bounded shape cache or
    shape buckets. Gradients keep their addresses between replays.
    """
    def __init__(self, model):
        self.model = model
        self.graph = None
        # AccumulateGrad remembers the stream on which its leaf was created.
        # Give capture fresh leaves on this same non-default stream, rather than
        # joining a previously used legacy stream during backward capture.
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for name, parameter in list(model.named_parameters()):
                parent = model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], torch.nn.Parameter(parameter.detach().clone(),
                                                             requires_grad=parameter.requires_grad))
        torch.cuda.current_stream().wait_stream(self.stream)
        import importlib
        module = importlib.import_module(type(model).__module__)
        tree = ast.parse(inspect.getsource(module))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SrsRWKV")
        loss = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_loss")
        loss.decorator_list = []
        loss = _CaptureLoss().visit(loss)
        tree = ast.fix_missing_locations(ast.Module(body=[loss], type_ignores=[]))
        namespace = dict(module.__dict__)
        exec(compile(tree, module.__file__, "exec"), namespace)
        self._loss = namespace["_get_loss"]

    def __getattr__(self, name):
        return getattr(self.model, name)

    def zero_grad(self, set_to_none=True):
        return self.model.zero_grad(set_to_none=False if self.graph is not None else set_to_none)

    @staticmethod
    def _tensors(batch):
        return (batch.start, *(tensor for group in batch.sub_gather for tensor in group),
                *(tensor for group in batch.time_shift_selects for tensor in group),
                *(tensor for group in batch.skips for tensor in group), batch.labels, batch.label_review_th)

    def _statistics(self):
        batch = self.static
        return self._loss(self.model, batch.start, batch.sub_gather, batch.sub_gather_lens,
                          batch.time_shift_selects, batch.skips, batch.num_data,
                          batch.labels, batch.label_review_th)

    def graph_update(self, batch, seed=2025):
        tensors = self._tensors(batch)
        shape = (batch.num_data, tuple(tuple(int(length) for length in group) for group in batch.sub_gather_lens),
                 tuple((tuple(tensor.shape), tensor.dtype) for tensor in tensors))
        if self.graph is None:
            self.static = batch
            self.shape = shape
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    self.model.zero_grad(set_to_none=True)
                    warm = self._statistics()
                    warm.average_loss.backward()
            torch.cuda.current_stream().wait_stream(self.stream)
            torch.cuda.synchronize()
            self.model.zero_grad(set_to_none=False)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.model.zero_grad(set_to_none=False)
                self.stats = self._statistics()
                self.stats.average_loss.backward()
            # Warmup/capture must not change this update's dropout seed.
            torch.manual_seed(seed)
        else:
            if self.shape != shape:
                raise ValueError("CUDA Graph benchmark encountered a different shape")
            for target, source in zip(self._tensors(self.static), tensors):
                target.copy_(source)
        self.graph.replay()
        return self.stats
