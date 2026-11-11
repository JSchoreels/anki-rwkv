#!/usr/bin/env python
"""The smaller instant-only model (SM101), the fork's "Tiny" RWKV-Instant model: finish the package
`artifacts/models/sm101` after `scripts/small_model_export.py package --arch w64-c1` (same exporter and options as the
sm72e25 / sm92 / sm98 packages, RWKV_DELETED_CARDS=neighbour) and `testvectors/make_testvectors.py` (the sm98 script with
the model built from architecture.json instead of a fixed w64: show-time query rows, the inputs SM101 was trained on):

  - neutral names as in the sm98 package: workspace paths -> <workspace>/..., the trainer folder `expertium` -> `run`,
    model name -> rwkv-small, optimizer expertium-iter53 -> muon-iter53;
  - manifest "evaluation": sm101-decision.json (722 users of the 732 set and the 1,500 validation users, instant head,
    show-time and normal inputs);
  - manifest "use": instant head only (its curve head is untrained: curve_loss_weight 0.0); "inputs": show-time intervals;
  - manifest "files" re-hashed.  Idempotent.

usage (from the worktree root): $PY -X utf8 artifacts/small-model/sm101/sm101_package.py
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "artifacts" / "models" / "sm101"
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
    text = text.replace('"artifacts/small-model/', '"<workspace>/wt/sm101/artifacts/small-model/')
    text = text.replace("/expertium/", "/run/").replace("expertium-iter53", "muon-iter53")
    text = text.replace('"expertium_catch_up"', '"catch_up"').replace('"expertium"', '"rwkv-small"')
    left = [m.group(0) for m in re.finditer(r".{0,40}expertium.{0,40}", text, re.I)]
    if left:
        raise SystemExit(f"names left: {left[:5]}")
    if "jonat" in text:
        raise SystemExit("a workspace path is left")
    return text


def head(rec):
    keep = ("new", "ref", "delta", "delta_ci", "points", "points_ci", "users_better", "users_worse")
    return {k: rec[k] for k in keep}


def evaluation():
    d = json.loads((HERE / "sm101-decision.json").read_text(encoding="utf-8"))
    names = {"new_minus_sm96_e2s": "new - the show-time instant-only small model (SM96), show-time inputs",
             "new_minus_sm98_e2s": "new - the show-time instant-only model, stage 2 (SM98), show-time inputs",
             "new_minus_release_e2s": "new - the release (SM63, sm72e25), show-time inputs",
             "new_minus_sm96_normal": "new - the show-time instant-only small model (SM96), normal inputs",
             "new_minus_sm98_normal": "new - the show-time instant-only model, stage 2 (SM98), normal inputs",
             "new_minus_sm92_normal": "new - the show-time retrain of the release (SM92), normal inputs",
             "new_e2s_minus_big_normal": "new on show-time inputs - the big model (1 leak-free epoch) on normal inputs",
             "new_e2s_minus_new_normal": "the new model's show-time cost (show-time minus normal inputs)"}
    pool = d["pool_1500"]
    return {
        "scoring": "leak-free inputs (deleted cards: neighbour), int8 per-channel weights + int8 card/note state (w8+s8), "
                   "review-weighted LogLoss, paired user bootstrap 2,000 resamples seed 0, 95% intervals",
        "inputs": {"e2s-query": "show-time intervals: every query asked before the answer uses the interval from the "
                                "card's previous answer to the moment the card is shown (what the Anki fork queries)",
                   "normal": "the published answer-to-answer intervals on every row"},
        "stage": "the show-time instant-only model, stage 2 (SM98)'s epoch 35 cut to one card layer (w64-c1) + 3 "
                 "constant-LR instant-only epochs (36-38) on show-time inputs + the release's one-epoch 32k Muon "
                 "cool-down on show-time inputs (curve_loss_weight 0.0)",
        "rule_pool_722": {"unit": "points = % of the big model (1 leak-free epoch) normal-input instant LogLoss "
                                  f"{d['big_instant']}, 722 users of the 732 set",
                          "rule": d["rule"], "verdict": d["verdict"],
                          "A new - the big model (1 leak-free epoch), normal inputs":
                              {"instant": head(d["A_bar_new_minus_big_normal"])},
                          "B new - the show-time retrain of the release (SM92), show-time inputs":
                              {"instant": head(d["B_new_minus_sm92_e2s"])},
                          **{names[k]: {"instant": head(v["instant"])} for k, v in d["secondary_732"].items()}},
        "validation_1500": {"unit": "% of the reference's LogLoss on the same users, show-time inputs",
                            "users": pool["users"],
                            "new - the release (SM63, sm72e25)": {"instant": head(pool["new_minus_release"]["instant"])},
                            "new - SM96": {"instant": head(pool["new_minus_sm96"]["instant"])},
                            "new - SM98": {"instant": head(pool["new_minus_sm98"]["instant"])}},
        "curve_head": "untrained (every curve loss term at weight 0; curve LogLoss "
                      f"+{pool['new_minus_release']['curve']['pct_of_ref']}% of the release's on the 1,500 validation "
                      "users): never use p_curve, intervals or S90 from this model",
    }


def main():
    parity = PACKAGE / "parity.json"
    parity.write_text(neutral(parity.read_text(encoding="utf-8")), encoding="utf-8")
    m = json.loads(neutral((PACKAGE / "manifest.json").read_text(encoding="utf-8")))
    m["evaluation"] = evaluation()
    m["use"] = {"heads": ["instant", "rating"],
                "rule": "instant-only model: trained with curve_loss_weight 0.0 and show_time_intervals from the "
                        "show-time instant-only model, stage 2 (SM98)'s epoch 35 cut to one card layer (epochs 36-38 + "
                        "the release's one-epoch 32k Muon cool-down). Its instant / rating head (p_imm, p_rating) is "
                        "the product; its curve head was never trained and must not schedule. In the Anki fork it is "
                        "the 'Tiny' RWKV-Instant model: alone without RWKV-Curve, or beside the release (sm72e25), "
                        "which keeps the curve, intervals, S90 and the answer-time head.",
                "architecture": "w64-c1: the small model (w64) with 1 card layer instead of 2 (stream card_id "
                                "n_layers 1, layer_steps without (0, 1)); every other stream and dimension unchanged"}
    m["inputs"] = {
        "show_time_intervals": True,
        "rule": "trained with the track option show_time_intervals: the instant query row of review j and the curve "
                "query for review j use floor(max(elapsed_seconds_j - duration_j / 1000, 0)) seconds (previous answer "
                "to the moment review j is shown), its sin / cos, and the cumulative interval = the card's earlier "
                "answer-to-answer intervals + that show-time interval; update rows keep the answer-to-answer interval "
                "and the cumulative interval including it. The test vectors are built this way (testvectors/"
                "make_testvectors.py, the held-out scorer's --inputs e2s-query rows), as for sm92 and sm98.",
        "leak_free": "RWKV_DELETED_CARDS=neighbour: missing-ID flags (features 13-15) always 0 (INTEGRATION.md 6.9)",
    }
    m["parity"] = json.loads(parity.read_text(encoding="utf-8"))
    m["files"] = {n: {"bytes": (PACKAGE / n).stat().st_size, "sha256": sha256(PACKAGE / n)} for n in FILES}
    (PACKAGE / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    print(f"[sm101-package] {PACKAGE}: manifest neutral, evaluation {m['evaluation']['rule_pool_722']['verdict']}, "
          f"{len(m['files'])} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
