"""SM58 follow-up: how much do the small and big model disagree? x = p_small - p_big per review.
Reports, per model / head / sample: reviews, share with |x| >= 0.025 (outside the central 5-point bin), >= 0.05, >= 0.10,
and percentiles p66 / p95 / p99 of |x|. Benchmark users from the SM58 dumps (same 722 users as summary.md);
own collection aggregate only (per-review predictions stay in collection-cache/, never committed).
  python sm58_disagreement_size.py  -> disagreement-size.{json,md}"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sm58_umplus as U

HERE = Path(__file__).resolve().parent
THRESH = (0.025, 0.05, 0.10)
PCTS = (66, 95, 99)


def stats(ax):
    ax = np.abs(ax)
    return {"reviews": int(len(ax)), **{f"share_ge_{t}": float((ax >= t).mean()) for t in THRESH},
            **{f"count_ge_{t}": int((ax >= t).sum()) for t in THRESH}, **{f"p{q}": float(np.percentile(ax, q)) for q in PCTS}}


def benchmark(tag, users, big_dirs):
    acc = {(h, s): [] for h in U.HEADS for s in U.SAMPLES}
    used = 0
    for u in users:
        sf = HERE / "dump" / f"{u}_{tag}_w8+s8.npz"
        bf = next((Path(d) / f"{u}_fork_w8+s8.npz" for d in big_dirs if (Path(d) / f"{u}_fork_w8+s8.npz").exists()), None)
        if not sf.exists() or bf is None:
            continue
        s, b = np.load(sf), np.load(bf)
        if not (np.array_equal(s["review_th"], b["review_th"]) and np.array_equal(s["y"], b["y"])):
            continue
        el = pd.read_parquet(U.ORIG / "revlogs" / f"user_id={u}", columns=["elapsed_seconds"])["elapsed_seconds"].to_numpy()
        elig = el[s["review_th"] - 1] >= 86400
        used += 1
        for h, (col, _, _) in U.HEADS.items():
            x = (s[col].astype(np.float64) - b[col].astype(np.float64)).astype(np.float32)
            acc[(h, "all")].append(x)
            acc[(h, "cross_day")].append(x[elig])
    return used, {f"{h}/{s}": stats(np.concatenate(v)) for (h, s), v in acc.items()}


def collection(tag):
    import io
    import lmdb
    import torch
    env = lmdb.open(str(U.EVAL / "collection-label-db"), readonly=True, lock=False)
    with env.begin() as txn:
        ths = torch.load(io.BytesIO(txn.get("1_review_ths".encode())), weights_only=True).tolist()
    env.close()
    revlogs = pd.read_parquet(U.EVAL / "collection-dataset" / "revlogs" / "user_id=1")
    revlogs["review_th"] = np.arange(1, len(revlogs) + 1)
    big = pd.read_parquet(U.EVAL / "preds-model-B-full-s0.parquet").rename(columns={"p_curve": "b_curve", "p_instant": "b_instant"})
    small = pd.read_parquet(HERE / "collection-cache" / f"preds-{tag}-full-s0.parquet").rename(columns={"p_curve": "a_curve", "p_instant": "a_instant"})
    d = (pd.DataFrame({"review_th": ths}).merge(big[["review_th", "b_curve", "b_instant"]], on="review_th")
         .merge(small[["review_th", "a_curve", "a_instant"]], on="review_th").merge(revlogs[["review_th", "elapsed_seconds"]], on="review_th"))
    assert len(d) == len(ths)
    out = {}
    for h in U.HEADS:
        x = (d[f"a_{h}"] - d[f"b_{h}"]).to_numpy()
        out[f"{h}/all"] = stats(x)
        out[f"{h}/cross_day"] = stats(x[(d.elapsed_seconds >= 86400).to_numpy()])
    return out


def main():
    users = json.loads((HERE / "users.json").read_text())["users"]
    big_dirs = [str(U.SM24_DUMP), str(HERE / "dump")]
    res = {}
    if (HERE / "disagreement-size.json").exists() and "--reuse" in __import__("sys").argv:
        res = json.loads((HERE / "disagreement-size.json").read_text())
    for tag in [t for t in ("sm36", "sm55") if t not in res]:
        n, b = benchmark(tag, users, big_dirs)
        res[tag] = {"benchmark_users": n, "benchmark": b, "collection": collection(tag)}
        print(tag, n, "users", flush=True)
    (HERE / "disagreement-size.json").write_text(json.dumps(res, indent=1) + "\n")
    names = {"sm36": "Muon version (SM36)", "sm55": "8k model (SM55)"}
    lines = ["# SM58 follow-up: how much the small and big model disagree", "",
             "x = p_small - p_big per review (w8+s8). Shares are of all test reviews in the sample; the central 5-point UM+ bin is |x| < 0.025. "
             "Own collection: aggregate only.", ""]
    for where, title in (("benchmark", "Benchmark users"), ("collection", "Own collection")):
        if where == "benchmark":
            title += f" ({res['sm36']['benchmark_users']} users)"
        lines += [f"## {title}", "", "| Model | Head | Sample | Reviews | |x| >= 0.025 | |x| >= 0.05 | |x| >= 0.10 | p66 |x| | p95 |x| | p99 |x| |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for tag in res:
            for key, s in res[tag][where].items():
                h, smp = key.split("/")
                lines.append(f"| {names[tag]} | {h} | {smp} | {s['reviews']:,} | {100 * s['share_ge_0.025']:.1f}% ({s['count_ge_0.025']:,}) | "
                             f"{100 * s['share_ge_0.05']:.1f}% | {100 * s['share_ge_0.1']:.1f}% | {s['p66']:.4f} | {s['p95']:.4f} | {s['p99']:.4f} |")
        lines.append("")
    (HERE / "disagreement-size.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
