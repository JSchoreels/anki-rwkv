"""SM148 factory page helper: close the plan row and insert the results row in pages/data/factory-curated.json by targeted
byte insertion (CRLF anchors, HTML entities). Run once from the main repo."""
import json
import re

WT = "C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148/"
P = "C:/Users/jonat/workspace/anki-rwkv/pages/data/factory-curated.json"
md = open(WT + "sm148-result.md", encoding="utf-8").read()
sec = md.split("## Rules per model")[1].split("## By answer")[0]
by = json.load(open(WT + "sm148-bytes.json"))["sizes"]
MINUS = "&minus;"


def m(s):
    return re.sub(r"(?<![\w\]])-(?=\d)", MINUS, s)


lab = {"Q4": "Q4: rank 4, int8 factors, every state",
       "T24": "T24: rank 2 below 16 answers, rank 4 from 16",
       "Q2": "Q2: rank 2, int8 factors, every state"}
rows, order = {}, []
for line in sec.splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) != 8 or c[1] not in lab:
        continue
    model, label, head, r2, r2t, r3, g, ok = c
    key = (model, label)
    if key not in rows:
        rows[key] = []
        order.append(key)
    rows[key].append((head, r2, r2t, r3, g, ok))


def size(model):
    return "Big" if model.startswith("Big") else "Small" if model.startswith("Small") else "Tiny"


trows = []
for (model, label) in order:
    r = rows[(model, label)]
    j = lambda i: " / ".join(m(x[i]) for x in r)
    name = (model.replace("six-layer Tiny two-head (SM130)", "the six-layer Tiny two-head model (SM130)")
            .replace("six-layer Tiny Instant-Only (SM130)", "the six-layer Tiny Instant-Only model (SM130)"))
    ok = r[0][5]
    verdict = "passes" if ok == "yes" else "fails" if ok == "no" else "passes the Tiny rule, fails the strict rule"
    pct = by[size(model)][label]["vs_int8_pct"]
    trows.append([name, lab[label], m(f"{pct:+.1f}%"), j(2) if size(model) == "Tiny" else "-", j(1), j(3),
                  " / ".join(x[4] for x in r), verdict])

row = {
    "date": "11-23", "kind": "gpu-score", "id": "SM148", "track": "small model",
    "name": "Low-rank int8 states (SM148): post-training rank 4, rank 2 then 4 and rank 2 card and note states with int8 factors against int8 rows, on the six release-candidate models",
    "eval": "The 732 set (723 users, show-time inputs, leak-free) scored on the GPU for the six release-candidate models (the show-time two-head and Instant-Only big models (SM91s2) and (SM99), the show-time two-head cool-down for Small (SM118) and the longer Small Instant-Only model (SM113), the six-layer Tiny two-head and Instant-Only models (SM130)), in four labels, cards and notes alike: int8 rows (w8+s8hx8), Q4 (rank 4), T24 (rank 2 below 16 answers, rank 4 from 16) and Q2 (rank 2), all with int8 factor vectors and float16 scales. The 50 long-card users scored on the CPU, every label directly, with a per-review and per-card |dR| read-out against int8. No training. Pre-registered rules: dataset-wide cost upper bound +0.05% per head (Tiny under the user&rsquo;s rule: within +1.00% of the matching Small at int8), cards with 128+ answers point +0.5% and upper bound +1.0%, at most 1% of cards with a review off by more than 0.10.",
    "fork": "Go picks: rank 4 (Q4) for Big and Small, rank 2 then 4 (T24) for Tiny under its rule (strict go: Q4). The GPU low-rank path is a torch step loop (0.28x to 0.46x of int8 speed), so the fork would need a fused kernel; status pending on Big coverage only, decision for the main session",
    "takeaway": "<strong>Go: rank 4 with int8 factors (Q4) for Big (&minus;74.4% state bytes vs int8) and Small (&minus;52.3%); for Tiny rank 2 below 16 answers then rank 4 (T24, &minus;64.7%) under the user&rsquo;s Tiny rule, strict go Q4 (&minus;51.1%).</strong> The best tiered int formats so far are &minus;43.5% / &minus;38.6% / &minus;42.7% (Big / Small / Tiny). Q4 costs at most +0.02% dataset-wide on every model and head with no card losing a real retention prediction (at most 0.0036% of cards with a review off by more than 0.10). T24 costs +0.03% to +0.09% (fails the strict +0.05% rule on instant heads) and Q2 +0.05% to +0.16% (fails). Tiny T24 stays within +0.51% (curve) and +0.60% to +0.66% (instant) of the matching Small at int8, under +1.00%. <strong>Pending on Big coverage only:</strong> both Big models have 718 of 720 users on the 732 set (5 users of 424k to 532k reviews run out of GPU memory even alone at cap 0.92) and 37 of 48 long-card users (13 users of 87k to 151k reviews exceed the 5.9 GB CPU memory limit at the scorer&rsquo;s peak; one process per model did not help); the margins for Q4 on Big are wide. <strong>Cost:</strong> GPU speed 0.38x (Big), 0.45x (Small), 0.28x (Tiny) of int8 with this unfused step loop; CPU 0.91x to 0.95x (Big), 0.92x to 0.95x (Small), 0.87x to 0.91x (Tiny).",
    "inputs": "show-time",
    "started_at": "2026-11-23T01:50+01:00", "ended_at": "2026-11-23T08:35+01:00",
    "gpu_minutes": 385, "cpu_minutes": 730, "tickets": 13,
    "detail": {
        "verdict": "Go (pending on Big coverage only): Q4 for Big (&minus;74.4% state bytes vs int8) and Small (&minus;52.3%), T24 for Tiny (&minus;64.7%; strict go Q4 &minus;51.1%). Branch sm148 (RESULT.md, sm148-result.md, sm148-result.json, sm148-dr.json, design.md)",
        "table": {
            "head": ["Model", "Label", "State bytes vs int8",
                     "Cost vs the matching Small model at int8, value [95% CI] (Tiny only; limit: upper bound +1.00%): curve / instant",
                     "Dataset-wide cost vs int8, value [95% CI] (limit: upper bound +0.05%): curve / instant",
                     "Cards with 128+ answers (CPU) vs int8 (limit: value +0.5%, upper bound +1.0%): curve / instant",
                     "Share of cards with a review |dR| &gt; 0.10 (long-card users; limit 1%): curve / instant",
                     "Verdict"],
            "rows": trows,
        },
        "notes": [
            "Coverage: Small and Tiny 723 of 723 GPU users and 50 of 50 long-card users per model and label. Big (both models): 718 of 720 rule users on the GPU and 37 of 48 long-card users. The 5 missing GPU users ran out of memory in five passes and in a further one-process-per-label ticket at cap 0.92, the highest allowed (13.0 GB allocated of 14.6 GB). The 13 missing long-card users were killed by the 5.9 GB memory guard at 6.0 to 7.3 GB, again when retried with one process per model (the peak is a single label&rsquo;s forward pass); the per-review read-out is incomplete for Big only. The int8 rows equal the int3 young tier (SM145) rows (largest difference 1e-6).",
            "Wall time: launched 01:50 Mon 11-23, report pending 08:35. GPU about 385 minutes in 13 tickets (Tiny 69, Small 100, Big 167 plus 26 of retries, smoke 15, coverage ticket 001503 about 5); CPU long-card chain about 730 process-minutes. Speed against int8, review evaluations per second on the GPU (732 set): Big 76,700 vs about 29,300 (0.38x), Small 124,600 vs about 57,000 (0.45x), Tiny 295,600 vs about 84,000 (0.28x); on the CPU (long-card users): Big 913 vs 832 to 866, Small 1,529 vs 1,404 to 1,451, Tiny 3,690 vs 3,205 to 3,369.",
        ],
    },
}
txt = json.dumps(row, indent=1, ensure_ascii=False)
txt = "\n".join(" " + l for l in txt.split("\n"))
new_row = (txt + ",\n").replace("\n", "\r\n")
b = open(P, "rb").read()
anchor = b' "results": [\r\n'
assert b.count(anchor) == 1
i = b.index(anchor) + len(anchor)
b = b[:i] + new_row.encode("utf-8") + b[i:]
s = b.index(b'  "id": "SM148"')
e = b.index(b'"id": "SM147"', s)
blk = b[s:e]
oldw = blk[blk.index(b'   "when": "Started 01:50'):]
oldw = oldw[:oldw.index(b"\r\n")]
neww = ('   "when": "Done 08:35 Mon 11-23 (status pending on Big coverage only): go rank 4 (Q4) for Big (&minus;74.4% state bytes vs int8) and Small (&minus;52.3%), rank 2 then 4 (T24) for Tiny (&minus;64.7%; strict go Q4 &minus;51.1%); Big has 718 of 720 GPU users and 37 of 48 long-card users (memory)",').encode()
blk = blk.replace(oldw, neww)
assert b'"state": "running"' in blk
blk = blk.replace(b'"state": "running"', b'"state": "done"')
old = b'"started_at": "2026-11-23T01:50+01:00"\r\n'
assert blk.count(old) == 1
blk = blk.replace(old, b'"started_at": "2026-11-23T01:50+01:00",\r\n   "ended_at": "2026-11-23T08:35+01:00"\r\n')
b = b[:s] + blk + b[e:]
json.loads(b.decode("utf-8"))
open(P, "wb").write(b)
for t in trows:
    print(t)
print("ok")
