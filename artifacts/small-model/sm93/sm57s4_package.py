#!/usr/bin/env python
"""The instant model for RWKV-Instant (SM93), stage 4: finish the package `artifacts/models/sm57s4` (the instant-only
small model (SM57), stage 4: 4 more instant-only epochs) after `scripts/small_model_export.py package` (same exporter and options as the
sm72e25 package, RWKV_DELETED_CARDS=neighbour) and `testvectors/make_testvectors.py` (normal inputs, the sm72e25 script):

  - neutral names as in the sm72e25 / sm92 packages: workspace paths -> <workspace>/..., the trainer folder
    `expertium` -> `run`, model name -> rwkv-small, optimizer expertium-iter53 -> muon-iter53;
  - manifest "evaluation": sm57-stage4-decision.json (the 1,500 validation users, candidate - the release, both heads) and the
    722-user rule pool numbers of the instant-only small model (SM57) (points of the big model (1 leak-free epoch));
  - manifest "use": instant head only (its curve head is untrained: curve_loss_weight 0.0); "inputs": normal inputs;
  - manifest "files" re-hashed.  Idempotent.

usage: $PY -X utf8 artifacts/small-model/sm93/sm93_package.py
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / "artifacts" / "models" / "sm57s4"
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
    text = text.replace('"artifacts/small-model/', '"<workspace>/wt/sm57/artifacts/small-model/')
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


SRC = Path("C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57")


def evaluation():
    p = json.loads((SRC / "sm57-stage4-pool1500.json").read_text(encoding="utf-8"))
    d = json.loads((SRC / "sm57-stage4-decision.json").read_text(encoding="utf-8"))
    out = {
        "scoring": "leak-free inputs (deleted cards: neighbour), NORMAL inputs (answer-to-answer intervals), int8 per-channel "
                   "weights + int8 card/note state (w8+s8), review-weighted LogLoss, paired user bootstrap 2,000 resamples "
                   "seed 0, 95% intervals",
        "inputs_note": "stage 4 was judged on normal inputs only; a show-time-input scoring of stage 1 and stage 4 is queued in "
                       "anki-rwkv (artifacts/small-model/showtime/)",
        "reference": "the released small model (leak-free, cool-down from epoch 25, package sm72e25)",
        "stage": "stage 4 = stage 1's instant-only epoch 26 + instant-only epochs 27-30 + the release's one-epoch 32k Muon cool-down",
        "validation_1500": {"unit": "% of the release's LogLoss on the same users", "users": p["users"],
                            "candidate_minus_release": {"instant": head(p["candidate_minus_release"]["instant"])},
                            "candidate_minus_stage1": {"instant": head(p["candidate_minus_stage1"]["instant"])}},
        "rule_pool_722": {"unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, 722 users of the 732 set",
                          "verdict": d.get("verdict"),
                          "stage4_minus_stage1": "-0.071 [-0.079, -0.064]",
                          "stage4_minus_release": "-0.233",
                          "stage4_minus_big_model": "+0.051"},
        "curve_head": "untrained (every curve loss term at weight 0): never use p_curve, intervals or S90 from this model",
    }
    return out


def main():
    parity = PACKAGE / "parity.json"
    parity.write_text(neutral(parity.read_text(encoding="utf-8")), encoding="utf-8")
    m = json.loads(neutral((PACKAGE / "manifest.json").read_text(encoding="utf-8")))
    m["evaluation"] = evaluation()
    m["use"] = {"heads": ["instant", "rating"],
                "rule": "instant-only model: trained with curve_loss_weight 0.0 from the release's epoch 25 (epochs 26-30 + "
                        "the release's one-epoch 32k Muon cool-down). Its instant / rating head (p_imm, p_rating) is "
                        "the product; its curve head was never trained (untrained: curve LogLoss +48.5% of the release's on 1,487 validation users) and must "
                        "not schedule. In the Anki fork it runs beside the release (sm72e25), which keeps the curve, "
                        "intervals, S90 and the answer-time head."}
    m["inputs"] = {"show_time_intervals": False,
                   "rule": "normal inputs (answer-to-answer intervals), as the release sm72e25; test vectors built by "
                           "the sm72e25 testvectors/make_testvectors.py",
                   "leak_free": "RWKV_DELETED_CARDS=neighbour: missing-ID flags (features 13-15) always 0 (INTEGRATION.md 6.9)"}
    m["parity"] = json.loads(parity.read_text(encoding="utf-8"))
    m["files"] = {n: {"bytes": (PACKAGE / n).stat().st_size, "sha256": sha256(PACKAGE / n)} for n in FILES}
    (PACKAGE / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    print(f"[sm57s4-package] {PACKAGE}: manifest neutral, stage 4 evaluation, {len(m['files'])} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
