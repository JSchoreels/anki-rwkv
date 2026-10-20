"""SM48 profile: split scorer wall time per user into model_seconds (forward + collect_predictions), the rest of the
loop (metrics / load_state_dict / writes) and preparation, from an existing heldout JSONL + its scorer log."""
import json, re, sys
from collections import defaultdict
rows_path, log_path, start_marker = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None
rows = [json.loads(l) for l in open(rows_path) if '"benchmark"' in l]
by_user = defaultdict(list)
for r in rows:
    by_user[r["user_id"]].append(r)
# wall per user from the log's cumulative elapsed seconds, per scorer invocation
wall, prev = {}, None
pat = re.compile(r"\[small-model\] (\d+)/(\d+) user (\d+) \((\d+) reviews\) \| (\d+)s")
inv = 0
for line in open(log_path, encoding="utf-8", errors="replace"):
    if "to score, grid" in line:
        prev, inv = 0, inv + 1
    m = pat.search(line)
    if m and prev is not None:
        e = int(m.group(5)); wall[(inv, int(m.group(3)))] = (e - prev, int(m.group(4))); prev = e
# sizes buckets
buckets = [(0, 10_000), (10_000, 50_000), (50_000, 150_000), (150_000, 400_000), (400_000, 10**7)]
print("bucket | users | reviews | wall s | model s (sum variants) | prep s | other s | model share | reviews/s wall")
tot = defaultdict(float)
for lo, hi in buckets:
    w = m_ = p = n = rv = 0
    for (inv_, u), (ws, revs) in wall.items():
        if not (lo <= revs < hi) or u not in by_user or inv_ < 2:
            continue
        rs = by_user[u]
        n += 1; rv += revs; w += ws
        m_ += sum(r["model_seconds"] for r in rs); p += rs[0]["prepare_seconds"]
    if n:
        print(f"{lo//1000}k-{hi//1000}k | {n} | {rv:,} | {w:.0f} | {m_:.0f} | {p:.0f} | {w-m_:.0f} | {m_/max(w,1):.2f} | {rv*len(rs)/max(w,1):,.0f}")
        for k, v in zip(("n", "rv", "w", "m", "p"), (n, rv, w, m_, p)):
            tot[k] += v
print(f"all | {tot['n']:.0f} | {tot['rv']:,.0f} | {tot['w']:.0f} | {tot['m']:.0f} | {tot['p']:.0f} | {tot['w']-tot['m']:.0f} | {tot['m']/tot['w']:.2f} | {tot['rv']*2/tot['w']:,.0f}")
