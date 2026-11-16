#!/usr/bin/env python
"""Finish the packages of the longer Small Instant-Only model (SM113) and the longer cool-down for Tiny (SM115) for the
Anki fork, after `scripts/small_model_export.py package` (run_packages.sh, RWKV_DELETED_CARDS=neighbour, CPU) and the
synthetic test vectors (testvectors/make_testvectors.py with RWKV_TESTVECTOR_DATA, users 900001 / 900002), as the
smaller instant-only model (SM101)'s package was finished (wt/sm101-package artifacts/small-model/sm101/sm101_package.py):
neutral names, manifest "evaluation" from the decision JSON, "use" instant head only, "inputs" show-time, "files"
re-hashed. Idempotent.

usage (worktree root): $PY -X utf8 artifacts/small-model/time-heads-package/instant_package.py sm113|sm115
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WT = Path("C:/Users/jonat/workspace/wt")
FILES = ("architecture.json", "master.pth", "parity.json", "weights.f32.bin", "weights.int8.bin",
         "testvectors/user_900001.npz", "testvectors/user_900002.npz")
MODELS = {
    "sm113": {"name": "the longer Small Instant-Only model (SM113)", "arch": "w64", "ref": "sm98",
              "ref_name": "the Small Instant-Only model (SM98)",
              "stage": "the show-time instant-only model, stage 2 (SM98)'s line distilled from the show-time "
                       "instant-only big model (SM99) as teacher, with a two-epoch 32k Muon cool-down in two passes "
                       "(curve_loss_weight 0.0, show-time inputs)",
              "fork": "the 'Small' RWKV-Instant model: alone without RWKV-Curve, or beside the show-time retrain "
                      "(SM92), which keeps the curve, intervals and S90",
              "architecture": "w64: the small model (5 streams, d_model 64)"},
    "sm115": {"name": "the longer cool-down for Tiny (SM115)", "arch": "w64-c1", "ref": "sm101",
              "ref_name": "the Tiny Instant-Only model (SM101)",
              "stage": "the Tiny Instant-Only model (SM101)'s line (one card layer) distilled from the show-time "
                       "instant-only big model (SM99) as teacher, with a two-epoch 32k Muon cool-down in two passes "
                       "(curve_loss_weight 0.0, show-time inputs)",
              "fork": "the 'Tiny' RWKV-Instant model: alone without RWKV-Curve, or beside the Tiny two-head model "
                      "(SM106), which keeps the curve, intervals and S90",
              "architecture": "w64-c1: the small model (w64) with 1 card layer instead of 2 (stream card_id n_layers "
                              "1, layer_steps without (0, 1)); every other stream and dimension unchanged"},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def neutral(text, key):
    text = re.sub(r"C:(?:\\\\|/)Users(?:\\\\|/)jonat(?:\\\\|/)workspace((?:(?:\\\\|/)[^\"\\\\/]+)*)",
                  lambda m: "<workspace>" + m.group(1).replace("\\\\", "/"), text)
    text = text.replace('"artifacts/small-model/', f'"<workspace>/wt/{key}/artifacts/small-model/')
    text = text.replace("/expertium/", "/run/").replace("expertium-iter53", "muon-iter53")
    text = text.replace('"expertium_catch_up"', '"catch_up"').replace('"expertium"', '"rwkv-small"')
    text = re.sub(r"expertium", "outside-research", text, flags=re.I)
    if "jonat" in text:
        raise SystemExit("a workspace path is left")
    return text


def head(rec):
    keep = ("new", "ref", "delta", "delta_ci", "points", "points_ci", "pct_of_ref", "pct_of_ref_ci", "users_better",
            "users_worse")
    return {k: rec[k] for k in keep if k in rec}


def evaluation(key):
    m = MODELS[key]
    d = json.loads((WT / key / "artifacts/small-model" / key / f"{key}-decision.json").read_text(encoding="utf-8"))
    primary = d[f"new_minus_{m['ref']}_e2s"]
    return {
        "scoring": "leak-free inputs (deleted cards: neighbour), int8 per-channel weights + int8 card/note state (w8+s8), "
                   "review-weighted LogLoss, paired user bootstrap 2,000 resamples seed 0, 95% intervals",
        "rule": d["rule"], "verdict": d["verdict"], "users": d["users"],
        "unit": f"points = % of the big model (1 leak-free epoch) normal-input instant LogLoss {d['big_instant']}",
        f"new - {m['ref_name']}, show-time inputs": {"instant": head(primary)},
        "secondary_732": {k: {"instant": head(v.get("instant", v))} for k, v in d["secondary_732"].items()},
        "validation_1500_users": d["pool_1500"].get("users"),
        "curve_head": "untrained (curve loss weight 0): never use p_curve, intervals or S90 from this model",
    }


def main(key):
    m = MODELS[key]
    package = ROOT / "artifacts" / "models" / key
    parity = package / "parity.json"
    parity.write_text(neutral(parity.read_text(encoding="utf-8"), key), encoding="utf-8")
    man = json.loads(neutral((package / "manifest.json").read_text(encoding="utf-8"), key))
    man["model"] = m["name"]
    man["evaluation"] = evaluation(key)
    man["use"] = {"heads": ["instant", "rating"],
                  "rule": f"instant-only model: {m['stage']}. Its instant / rating head (p_imm, p_rating) is the "
                          f"product; its curve head was never trained and must not schedule. In the Anki fork it is "
                          f"{m['fork']}.",
                  "architecture": m["architecture"]}
    man["inputs"] = {
        "show_time_intervals": True,
        "rule": "trained with show-time intervals: the instant query row of review j and the curve query for review j "
                "use floor(max(elapsed_seconds_j - duration_j / 1000, 0)) seconds (previous answer to the moment "
                "review j is shown), its sin / cos, and the cumulative interval = the card's earlier answer-to-answer "
                "intervals + that show-time interval; update rows keep the answer-to-answer interval. The test vectors "
                "are built this way (testvectors/make_testvectors.py, the held-out scorer's --inputs e2s-query rows).",
        "leak_free": "RWKV_DELETED_CARDS=neighbour: missing-ID flags (features 13-15) always 0",
    }
    man["testvectors"] = {"users": [900001, 900002],
                          "source": "synthetic users (no dataset data): scripts/synthetic_testvector_users.py, read "
                                    "through RWKV_TESTVECTOR_DATA; testvectors/check_testvectors.py PASS"}
    man["parity"] = json.loads(parity.read_text(encoding="utf-8"))
    man["files"] = {n: {"bytes": (package / n).stat().st_size, "sha256": sha256(package / n)} for n in FILES}
    text = json.dumps(man, indent=1, default=str)
    if re.search("expertium", text, re.I) or "jonat" in text:
        raise SystemExit("names left in the manifest")
    (package / "manifest.json").write_text(text, encoding="utf-8")
    print(f"[instant-package] {package}: verdict {man['evaluation']['verdict']}, {len(man['files'])} files hashed")


if __name__ == "__main__":
    main(sys.argv[1])
