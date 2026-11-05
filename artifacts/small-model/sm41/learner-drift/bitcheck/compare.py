"""The changing-learner simulation (SM41g): default rows of the new engine vs the rows of the code before the change
(policy/bitcheck/after-policies.jsonl, same user 1606 split 660, 9 default lanes); wall-clock fields excluded."""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
WALL = {"wall_seconds", "lane_reviews_per_second", "warmup_seconds", "finished"}
ref = json.loads((H.parents[1] / "policy" / "bitcheck" / "after-policies.jsonl").read_text().splitlines()[0])
for name in ("default", "ivlhist"):
    r = json.loads((H / f"{name}.jsonl").read_text().splitlines()[0])
    if name == "ivlhist":
        r["acc"].pop("ivl_later"); r.pop("ivl_edges")
    a = {k: v for k, v in ref.items() if k not in WALL}
    b = {k: v for k, v in r.items() if k not in WALL}
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    print(f"{name}: {'IDENTICAL' if not diff else 'DIFFERS ' + str(diff)} ({len(a)} fields compared)")
