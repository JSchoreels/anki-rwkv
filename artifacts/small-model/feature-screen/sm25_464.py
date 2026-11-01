"""SM25 on the 464 users: students vs the normal fork and vs the fork with no-steps review-type labels (sm25.py),
review-weighted LogLoss in % (bar P style), on the users all labels have."""
import sys
from pathlib import Path

SM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SM.parents[1] / "scripts"))
import small_model_gate as g  # noqa: E402

STUDENTS = ["s-w64kd-gate-ctx32768c-dec10/w8+s8", "s-w64kd-gate-ctx16384c-dec9-L32k/w8+s8"]
per = g.load(SM / "heldout.jsonl", ["fork/w8+s8"] + STUDENTS, set(range(10 ** 6)))
per.update(g.load(SM / "sm25-scores.jsonl", ["forknosteps/w8+s8"], set(range(10 ** 6))))
for st in STUDENTS:
    users = sorted(set(per[st]) & set(per["fork/w8+s8"]) & set(per["forknosteps/w8+s8"]))
    if not users:
        print(f"{st}: no common users"); continue
    cells = []
    for h, name in g.HEADS.items():
        v = {k: g.rw(*g.weighted(per, k, users, h)) for k in ("fork/w8+s8", "forknosteps/w8+s8", st)}
        cells.append(f"{name}: student vs fork {100 * (v[st] / v['fork/w8+s8'] - 1):+.2f}%, vs no-steps fork "
                     f"{100 * (v[st] / v['forknosteps/w8+s8'] - 1):+.2f}% (no-steps fork vs fork {100 * (v['forknosteps/w8+s8'] / v['fork/w8+s8'] - 1):+.2f}%)")
    print(f"{st} on {len(users)} users | " + " | ".join(cells))
