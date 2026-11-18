#!/usr/bin/env python
"""Show-time retrain of the release (SM92): finish the release package `artifacts/models/sm92` after
`scripts/small_model_export.py package` (same exporter and options as the sm72e25 package, RWKV_DELETED_CARDS=neighbour)
and `testvectors/make_testvectors.py` (show-time query rows):

  - neutral names as in the sm72e25 package: paths under the workspace -> <workspace>/..., the trainer folder
    `expertium` -> `run`, model name -> rwkv-small, optimizer expertium-iter53 -> muon-iter53;
  - manifest "evaluation": the decision rows of artifacts/small-model/sm92/sm92-decision.json (722 users of the 732 set,
    w8+s8, leak-free, show-time inputs `e2s-query` and normal inputs), "inputs": the show-time interval rule;
  - manifest "files" re-hashed (the same top-level files as the sm72e25 manifest lists).

usage: $PY -X utf8 sm92_package.py [--package DIR]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SM92 = HERE.parent
PACKAGE = SM92.parents[1] / "models" / "sm92"
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
    text = text.replace('"artifacts/small-model/', '"<workspace>/wt/sm92/artifacts/small-model/')
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
    d = json.loads((SM92 / "sm92-decision.json").read_text(encoding="utf-8"))
    pools = {}
    for name, key in (("732 set (722 users)", "rule_pool"), ("732+long set (725 users)", "732+long")):
        pools[name] = {cmp: {h: head(v[h]) for h in ("curve", "instant")}
                       for cmp, v in d[key].items() if isinstance(v, dict) and "curve" in v}
    return {
        "scoring": "leak-free inputs (deleted cards: neighbour), int8 per-channel weights + int8 card/note state (w8+s8), "
                   "review-weighted LogLoss, paired user bootstrap 2,000 resamples seed 0, 95% intervals; points = % of "
                   "the big model (1 leak-free epoch) normal-input LogLoss",
        "inputs": {"e2s-query": "show-time intervals: every query asked before the answer uses the interval from the "
                                "card's previous answer to the moment the card is shown (what the Anki fork queries)",
                   "normal": "the published answer-to-answer intervals on every row"},
        "reference": "the released small model (leak-free, cool-down from epoch 25, package sm72e25)",
        "verdict": d["verdict"], "why": d["why"], "confound": d["confound"], "rule": d["rule"],
        "comparisons": pools,
    }


def main():
    package = Path(sys.argv[sys.argv.index("--package") + 1]) if "--package" in sys.argv else PACKAGE
    parity = package / "parity.json"
    parity.write_text(neutral(parity.read_text(encoding="utf-8")), encoding="utf-8")
    m = json.loads(neutral((package / "manifest.json").read_text(encoding="utf-8")))
    m["evaluation"] = evaluation()
    m["inputs"] = {
        "show_time_intervals": True,
        "rule": "trained with the track option show_time_intervals: the instant query row of review j and the curve "
                "query for review j use floor(max(elapsed_seconds_j - duration_j / 1000, 0)) seconds (previous answer "
                "to the moment review j is shown), its sin / cos, and the cumulative interval = the card's earlier "
                "answer-to-answer intervals + that show-time interval; update rows keep the answer-to-answer interval "
                "and the cumulative interval including it. The test vectors are built this way (testvectors/"
                "make_testvectors.py, the held-out scorer's --inputs e2s-query rows).",
        "leak_free": "RWKV_DELETED_CARDS=neighbour: missing-ID flags (features 13-15) always 0 (INTEGRATION.md 6.9)",
    }
    m["parity"] = json.loads(parity.read_text(encoding="utf-8"))
    m["files"] = {n: {"bytes": (package / n).stat().st_size, "sha256": sha256(package / n)} for n in FILES}
    (package / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    print(f"[sm92-package] {package}: manifest neutral, evaluation {m['evaluation']['verdict']}, "
          f"{len(m['files'])} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
