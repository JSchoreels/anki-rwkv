"""CPU check of SM40v2: state_carry {"lanes": "strided"} with ONE lane (batch_chunks 1: one window per update).

Same setup as scripts/check_state_carry_lanes.py (users 1340 and 8254, w64 e6 checkpoint, float32, CPU eager, windows
of 2,048 standing in for the production 32,768 x 1). No GPU, no trainer change needed. Checks:
  L1  layout: strided_layout(N, 1) = one window per update in order, item.carry_lanes == [0], offsets / reviews / seeds
      equal the v1 (lanes "update") items at batch_chunks 1, window rows equal the full-history rows;
  B1  first update (lane start, empty table), train mode: loss and every gradient bitwise equal to stock get_loss;
  D1  carry through the trainer path (fp32 table, eval): every window of every update matches the user's single
      full-history forward (|dp|, |dw| < 1e-4), i.e. each window starts from its predecessor's exact end state;
  V1  the same predictions equal the v1 carry at batch_chunks 1 (|dp| < 1e-6).
  .venv/Scripts/python.exe -X utf8 scripts/check_state_carry_lane1.py [--users 1340,8254]
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_state_carry_trainer import CKPT, DATA, RUN, tensors_digest  # noqa: E402


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--users", default="1340,8254")
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-state-carry-lane1.json"))
    args = parser.parse_args()
    os.environ["RWKV_NO_JIT"] = "1"
    os.environ["GPU_BATCH1_DATA_PATH"] = DATA
    import torch
    torch.set_num_threads(4)
    from concurrent.futures import ThreadPoolExecutor
    from expertium.variants import build
    from rwkv.data_processing import get_rwkv_data
    from scripts.check_state_carry_lanes import _full_items
    from scripts.expertium_options import CatchUp
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_r01_prefetch import training_batches
    from scripts.gpu_r01_samples import samples_for
    from scripts.state_carry_train import strided_layout, user_segment_features

    started = time.perf_counter()
    users = [int(u) for u in args.users.split(",")]
    protocol = json.loads((RUN / "run" / "epoch-01" / "protocol.json").read_text())
    run_options = dict(json.loads((RUN / "experiment.json").read_text())["options"])
    plain = {k: v for k, v in run_options.items() if k not in ("packed_ipc", "distillation")}
    rows = [next(r for r in protocol["rows"] if r["user_id"] == u) for u in users]
    W = plain["chunk_reviews"]
    model = build("w64").float()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
    report = {"users": users, "window": W, "checks": {}}
    fails = []

    def opts(spec):
        return dict(plain, batch_chunks=1, state_carry=spec)

    def items(options, user_rows=rows):
        with ThreadPoolExecutor(max_workers=1) as pool:
            prepared = list(training_batches(user_rows, W, options["training_seed"], pool, options, None, 2))
        return [(row, item, item.batch.to(torch.device("cpu"))) for row, item in prepared]

    frames = {row["user_id"]: user_segment_features(split_frame(get_rwkv_data(Path(DATA), row["user_id"]), row, "train"))
              for row in rows}
    full = {}
    model.eval()
    for row in rows:
        user = row["user_id"]
        reviews = len(frames[user])
        full_options = dict(opts({"precision": "fp32"}), chunk_reviews=reviews)
        (_, _, batch), = _full_items(training_batches, ThreadPoolExecutor, full_options, reviews, row)
        with torch.no_grad():
            out = model.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens,
                                              batch.time_shift_selects, batch.skips, batch.num_data)
        (sample,) = samples_for(frames[user], user, full_options)
        keys = (sample.review_ths.long() * 2 + sample.skips.long()).tolist()
        full[user] = {"p": torch.softmax(out[3].float(), dim=-1)[0], "w": out[1].float()[0],
                      "where": {k: i for i, k in enumerate(keys)}, "windows": -(-reviews // W)}

    def index_of(options, user, window):
        (sample,) = samples_for(frames[user].iloc[window * W:(window + 1) * W], user, options)
        keys = (sample.review_ths.long() * 2 + sample.skips.long()).tolist()
        return torch.tensor([full[user]["where"][k] for k in keys])

    spec2 = {"precision": "fp32", "lanes": "strided"}
    v2 = items(opts(spec2))
    v1 = items(opts({"precision": "fp32"}))

    # L1
    l_rows = []
    for row in rows:
        user = row["user_id"]
        twos = [x for x in v2 if x[0]["user_id"] == user]
        ones = [x for x in v1 if x[0]["user_id"] == user]
        layout = strided_layout(full[user]["windows"], 1)
        l_rows.append({"user": user, "updates": len(twos), "windows": full[user]["windows"],
                       "layout_one_window_each": layout == [[(0, u)] for u in range(full[user]["windows"])],
                       "carry_lanes_zero": all(list(i.carry_lanes) == [0] for _, i, _ in twos),
                       "meta_equals_v1": [(i.offset, i.reviews, i.seed) for _, i, _ in twos] == [(i.offset, i.reviews, i.seed) for _, i, _ in ones],
                       "start_equals_v1_batches": all(torch.equal(a[2].start, b[2].start) for a, b in zip(twos, ones))})
    ll = all(all(v for k, v in r.items() if k not in ("user", "updates", "windows")) and r["updates"] == r["windows"] > 1 for r in l_rows)
    report["checks"]["L1_layout"] = {"pass": ll, "rows": l_rows}
    fails += [] if ll else ["L1"]

    # B1
    b_rows = []
    for row, item, batch in v2:
        if item.offset != 0:
            continue
        results = []
        for carried in (False, True):
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(item.seed)
            if carried:
                catch_up = CatchUp(opts(spec2), {}, rows)
                catch_up.carry.attach(item, batch, row["user_id"])
                stats = catch_up.loss(model, batch, 0, (row["user_id"], item.offset, item.reviews, item.seed))
            else:
                stats = model.get_loss(batch)
            stats.average_loss.backward()
            results.append(tensors_digest([stats.average_loss, *[p.grad for p in model.parameters()]]))
        b_rows.append({"user": row["user_id"], "equal": results[0] == results[1]})
    b = len(b_rows) == len(users) and all(r["equal"] for r in b_rows)
    report["checks"]["B1_first_update_equals_stock_bitwise"] = {"pass": b, "rows": b_rows}
    fails += [] if b else ["B1"]
    model.zero_grad(set_to_none=True)
    model.eval()

    # D1 / V1
    d_rows = []
    for row in rows:
        user = row["user_id"]
        cu2, cu1 = CatchUp(opts(spec2), {}, [row]), CatchUp(opts({"precision": "fp32"}), {}, [row])
        worst = {"dp_full": 0.0, "dw_full": 0.0, "dp_v1": 0.0, "dp_windows_after_first": 0.0}
        pairs = zip([x for x in v2 if x[0]["user_id"] == user], [x for x in v1 if x[0]["user_id"] == user])
        for (_, i2, b2), (_, i1, b1) in pairs:
            cu2.carry.attach(i2, b2, user)
            cu1.carry.attach(i1, b1, user)
            with torch.no_grad():
                s2, s1 = cu2.loss(model, b2, 0), cu1.loss(model, b1, 0)
            window = i2.offset // W
            index = index_of(opts(spec2), user, window)
            n = index.numel()
            dp = float((s2.p_imm_all[0, :n].float() - full[user]["p"].index_select(0, index)).abs().max())
            dw = float((s2.w[0, :n].float() - full[user]["w"].index_select(0, index)).abs().max())
            worst["dp_full"], worst["dw_full"] = max(worst["dp_full"], dp), max(worst["dw_full"], dw)
            if window > 0:
                worst["dp_windows_after_first"] = max(worst["dp_windows_after_first"], dp)
            worst["dp_v1"] = max(worst["dp_v1"], float((s2.p_imm_all - s1.p_imm_all).abs().max()))
        d_rows.append({"user": user, "windows": full[user]["windows"], **worst})
    d = all(r["dp_full"] < 1e-4 and r["dw_full"] < 1e-4 and r["windows"] > 2 for r in d_rows)
    v = all(r["dp_v1"] < 1e-6 for r in d_rows)
    report["checks"]["D1_every_window_matches_full_history"] = {"pass": d, "rows": d_rows}
    report["checks"]["V1_equals_v1_carry_at_one_window_per_update"] = {"pass": v}
    fails += ([] if d else ["D1"]) + ([] if v else ["V1"])

    report["fails"], report["passed"] = fails, not fails
    report["seconds"] = round(time.perf_counter() - started, 1)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v.get("pass") for k, v in report["checks"].items()}))
    print("ALL OK" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
