#!/usr/bin/env python
"""The Tiny two-head model (SM106), the fork's Tiny one-model choice: finish the package `artifacts/models/sm106`
after `scripts/small_model_export.py package --arch w64-c1` (the exporter and options of the sm92 / sm98 / sm101
packages, RWKV_DELETED_CARDS=neighbour, CPU) and `testvectors/make_testvectors.py` (the sm101 script: show-time query
rows, `--inputs e2s-query`, the inputs SM106 was trained on; inputs identical to sm101's vectors):

  - neutral names as in the sm101 package: workspace paths -> <workspace>/..., the trainer folder `expertium` -> `run`,
    model name -> rwkv-small, optimizer expertium-iter53 -> muon-iter53;
  - manifest "evaluation": sm106-decision.json (722 users of the 732 set and the 1,487 validation users, both heads,
    show-time inputs);
  - manifest "use": both heads (curve, instant, rating): unlike the Tiny Instant-Only model (SM101) its curve head is
    trained, so p_curve, intervals and the 90% stability may be used, as for the show-time retrain (SM92);
    "inputs": show-time intervals;
  - manifest "files" re-hashed.  Idempotent.

usage (from the worktree root): $PY -X utf8 artifacts/small-model/sm106-package/sm106_package.py
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "artifacts" / "models" / "sm106"
FILES = ("architecture.json", "master.pth", "parity.json", "weights.f32.bin", "weights.int8.bin")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def neutral(text):
    text = re.sub(r"C:(?:\\\\|/)Users(?:\\\\|/)jonat(?:\\\\|/)workspace((?:(?:\\\\|/)[^\"\\\\/]+)*)",
                  lambda m: "<workspace>" + m.group(1).replace("\\\\", "/"), text)
    text = text.replace('"artifacts/small-model/', '"<workspace>/wt/sm106/artifacts/small-model/')
    text = text.replace("/expertium/", "/run/").replace("expertium-iter53", "muon-iter53")
    text = text.replace('"expertium_catch_up"', '"catch_up"').replace('"expertium"', '"rwkv-small"')
    left = [m.group(0) for m in re.finditer(r".{0,40}expertium.{0,40}", text, re.I)]
    if left:
        raise SystemExit(f"names left: {left[:5]}")
    if "jonat" in text:
        raise SystemExit("a workspace path is left")
    return text


def head(rec):
    keep = ("new", "ref", "delta", "delta_ci", "pct_of_ref", "pct_of_ref_ci", "users_better", "users_worse")
    return {k: rec[k] for k in keep}


def heads(rec, names=("curve", "instant")):
    return {n: head(rec[n]) for n in names}


def evaluation():
    d = json.loads((HERE / "sm106-decision.json").read_text(encoding="utf-8"))
    b, pool = d["B_report"], d["pool_1500"]
    return {
        "scoring": "leak-free inputs (deleted cards: neighbour), int8 per-channel weights + int8 card/note state (w8+s8), "
                   "review-weighted LogLoss, paired user bootstrap 2,000 resamples seed 0, 95% intervals",
        "inputs": {"e2s-query": "show-time intervals: every query asked before the answer uses the interval from the "
                                "card's previous answer to the moment the card is shown (what the Anki fork queries)"},
        "stage": "the show-time retrain (SM92)'s constant-LR epoch 26 cut to one card layer (w64-c1) + 3 constant-LR "
                 "epochs (27-29) on show-time inputs with both heads at SM92's loss weights + the release's one-epoch "
                 "32k Muon cool-down on show-time inputs (both heads)",
        "rule_pool_722": {
            "unit": "% of the big model (1 leak-free epoch)'s show-time LogLoss on the same users (curve "
                    f"{d['big_e2s']['curve']}, instant {d['big_e2s']['instant']}), 722 users of the 732 set",
            "rule": d["rule"], "verdict": d["verdict"], "pass": d["A_pass"],
            "A new - the big model (1 leak-free epoch), show-time inputs": heads(d["A_new_minus_big_e2s"]),
            "new - the show-time retrain (SM92), show-time inputs": heads(b["new_minus_sm92_e2s"]),
            "new - the release (SM63, sm72e25), show-time inputs": heads(b["new_minus_release_e2s"]),
            "new - the Tiny Instant-Only model (SM101), show-time inputs":
                heads(b["new_minus_sm101_e2s"], ("instant",)),
        },
        "validation_1500": {
            "unit": "% of the reference's own LogLoss on the same users, show-time inputs",
            "users": pool["users"],
            "new - the release (SM63, sm72e25)": heads(pool["new_minus_release"]),
            "new - the Tiny Instant-Only model (SM101)": heads(pool["new_minus_sm101"], ("instant",)),
            "row_check": pool["row_check"],
        },
    }


def main():
    parity = PACKAGE / "parity.json"
    parity.write_text(neutral(parity.read_text(encoding="utf-8")), encoding="utf-8")
    m = json.loads(neutral((PACKAGE / "manifest.json").read_text(encoding="utf-8")))
    m["evaluation"] = evaluation()
    m["use"] = {"heads": ["curve", "instant", "rating"],
                "rule": "two-head model: trained with both heads at the show-time retrain (SM92)'s loss weights and "
                        "show_time_intervals, from SM92's epoch 26 cut to one card layer (epochs 27-29 + the "
                        "release's one-epoch 32k Muon cool-down). Its curve head (p_curve, intervals, S90) and its "
                        "instant / rating head (p_imm, p_rating) are both the product. It has no answer-time head "
                        "(none was trained on its trunk). In the Anki fork it is the Tiny size's one-model choice: "
                        "alone for both schedulers when the one-model opt-out is on.",
                "architecture": "w64-c1: the small model (w64) with 1 card layer instead of 2 (stream card_id "
                                "n_layers 1, layer_steps without (0, 1)); every other stream and dimension unchanged"}
    m["inputs"] = {
        "show_time_intervals": True,
        "rule": "trained with the track option show_time_intervals: the instant query row of review j and the curve "
                "query for review j use floor(max(elapsed_seconds_j - duration_j / 1000, 0)) seconds (previous answer "
                "to the moment review j is shown), its sin / cos, and the cumulative interval = the card's earlier "
                "answer-to-answer intervals + that show-time interval; update rows keep the answer-to-answer interval "
                "and the cumulative interval including it. The test vectors are built this way (testvectors/"
                "make_testvectors.py, the held-out scorer's --inputs e2s-query rows), as for sm92, sm98 and sm101.",
        "leak_free": "RWKV_DELETED_CARDS=neighbour: missing-ID flags (features 13-15) always 0 (INTEGRATION.md 6.9)",
    }
    m["parity"] = json.loads(parity.read_text(encoding="utf-8"))
    m["files"] = {n: {"bytes": (PACKAGE / n).stat().st_size, "sha256": sha256(PACKAGE / n)} for n in FILES}
    (PACKAGE / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    print(f"[sm106-package] {PACKAGE}: manifest neutral, evaluation {m['evaluation']['rule_pool_722']['verdict']}, "
          f"{len(m['files'])} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
