"""SM148 verdict under the user's 11-23 budgets (set after the data existed). CPU only, reuses sm148_report helpers
(showtime.compare paired per user; the reference rows passed are the Big model's)."""
import json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import sm148_report as R
st = R.showtime()
rows, _, _ = R.read_rows(R.ROWS732)
U = set(json.loads(R.U732.read_text())["users"]) if isinstance(json.loads(R.U732.read_text()), dict) else set(json.loads(R.U732.read_text()))
lab = lambda k, s: f"{R.MODELS[k][0]}/{R.LABELS[s]}{R.E2S}"
get = lambda k, s: R.sub(rows, lab(k, s), U)
BUD = {"Big": 0.15, "Small": 0.5, "Tiny": 1.0}
REFM = {"sm91s2": "sm91s2", "sm99": "sm99", "sm118": "sm91s2", "sm113": "sm99", "sm130th": "sm91s2", "sm130io": "sm99"}
LABS = ["REF", "Q4", "T24", "Q2"]
out = {}
for k, (tag, name, heads, size, _) in R.MODELS.items():
    for s in LABS:
        new = get(k, s)
        ref = get(REFM[k], "REF") if size != "Big" else get(k, "REF")
        if size == "Big" and s == "REF":
            continue
        common = sorted(set(new) & set(ref))
        out[(k, s)] = (R.dataset_cmp(new, ref, heads, common, st), len(common))
res = json.loads(R.HERE.joinpath("sm148-result.json").read_text())
tab = []; go = {}
for k, (tag, name, heads, size, _) in R.MODELS.items():
    for s in LABS:
        if (k, s) not in out: continue
        c, n = out[(k, s)]
        for h in heads:
            up = c[h]["ci"][1]
            tab.append(dict(model=name, size=size, label=s, head=h, users=n, value=c[h]["value"], lo=c[h]["ci"][0], up=up,
                            budget=BUD[size], margin=BUD[size] - up, ok=up <= BUD[size]))
byt = R.bytes_rank()
for size in BUD:
    ks = [k for k, v in R.MODELS.items() if v[3] == size]
    ok = {s: all(r["ok"] for r in tab if r["size"] == size and r["label"] == s) for s in LABS if not (size == "Big" and s == "REF")}
    ok["REF"] = True if size == "Big" else ok["REF"]
    order = sorted(LABS, key=lambda s: byt[size][s])  # most negative first
    pick = next((s for s in order if ok[s]), None)
    worst = max(r["up"] for r in tab if r["size"] == size and r["label"] == pick)
    go[size] = dict(pick=pick, ok=ok, bytes=byt[size][pick], worst_up=worst, budget=BUD[size], margin=BUD[size] - worst)
    # warnings
    w = []
    for k in ks:
        m = res["models"][k]["schemes"].get(pick)
        if m: w.append((k, m["R3_long_128p"], m["G_card_share_gt_0.10"]))
    go[size]["warn"] = w
json.dump(dict(table=tab, go=go), open(R.HERE / "sm148-verdict-1123.json", "w"), indent=1, default=str)
for r in tab: print(f"{r['size']:5} {r['model']:38} {r['label']:3} {r['head']:7} n={r['users']} {r['value']:+.3f} [{r['lo']:+.3f},{r['up']:+.3f}] bud {r['budget']} margin {r['margin']:+.3f} {'ok' if r['ok'] else 'FAIL'}")
for s, g in go.items(): print(s, g["pick"], g["bytes"], g["worst_up"], g["margin"], g["ok"])
for s, g in go.items():
    for k, r3, gg in g["warn"]:
        print(s, k, g["pick"], {h: (round(v["value"],3), round(v["ci"][1],3)) if v else None for h, v in r3.items()}, gg)
