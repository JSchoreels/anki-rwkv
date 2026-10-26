"""Rebuild the fork's RWKV "Memorised" series (Cards mode) on a copy of the user's collection, CPU only, and
decompose each day's change.

The fork (qt/aqt/rwkv_scheduler.py `_compute_rwkv_memorised_history`) warms up each Anki day's reviews, then for every
card seen so far runs one *query* step with the card's last review input (is_query, no rating/duration, elapsed =
query day - last review day, in whole days) through the card/deck/note/preset/global states, and sums the instant
head's 1 - P(Again). This script mirrors that with this repo's upstream `rwkv/` RNN code and the fork's weights
(Model B, `RWKV_trained_on_5000_10000.pth`), on the converted collection of 2026-10-14
(artifacts/collection-eval/2026-10-14/collection-dataset).

For each query day d it records per card (scratch only, never committed):
  full  R(states_d, tracker_d)                        the fork's value
  q0    R(states_{d-1}, tracker_{d-1}) at day d       "no review today" counterfactual: pure time effect
  qg    q0 but with the global state of day d         + shared global-state shift
  qs    q0 but with every state of day d              + deck/preset/note state shifts
  (full - qs) = the tracker features (diff_reviews, diff_new_cards, day_offset_diff, cum_reviews_today ...).
Output: <scratch>/replay.npz (per-card arrays, aggregate-only use) and progress lines.
"""
import argparse
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--scratch", required=True)
parser.add_argument("--query-start", default="2025-09-01", help="first calendar day queried")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--ckpt-days", type=int, default=30, help="checkpoint (resumable) every N query days")
parser.add_argument("--resume", action="store_true", help="continue from <scratch>/ckpt.pt")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = str(args.threads)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import ctypes  # noqa: E402
import datetime as dt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG  # noqa: E402
from rwkv.config import DAY_OFFSET_ENCODE_PERIODS, ID_ENCODE_DIMS, ID_SPLIT  # noqa: E402
from rwkv.data_processing import (  # noqa: E402
    CARD_FEATURE_COLUMNS, get_rwkv_data, scale_cum_new_cards_today, scale_cum_reviews_today,
    scale_day_offset_diff, scale_diff_new_cards, scale_diff_reviews, scale_elapsed_days,
    scale_elapsed_days_cumulative, scale_elapsed_seconds, scale_elapsed_seconds_cumulative)
from rwkv.model.srs_model_rnn import SrsRWKVRnn  # noqa: E402

if os.name == "nt":
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
torch.set_num_threads(args.threads)
assert not torch.cuda.is_available()

EVAL = ROOT / "artifacts/collection-eval/2026-10-14"
MODEL_B = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")
SCRATCH = Path(args.scratch)
SCRATCH.mkdir(parents=True, exist_ok=True)

frame = get_rwkv_data(EVAL / "collection-dataset", 1).reset_index(drop=True)
assert frame.review_th.is_monotonic_increasing and frame.day_offset.is_monotonic_increasing
ids = pd.read_parquet(EVAL / "review-th-to-revlog-id.parquet").set_index("review_th").revlog_id
ts = frame.review_th.map(ids).to_numpy()
# Anki day = local date of (time - 6 h rollover)
first_date = (dt.datetime.fromtimestamp(ts[0] / 1000) - dt.timedelta(hours=6)).date()
day0 = int(frame.day_offset.iloc[0])
dates_check = [(dt.datetime.fromtimestamp(t / 1000) - dt.timedelta(hours=6)).date() for t in ts[::5000]]
offs_check = frame.day_offset.to_numpy()[::5000]
assert all((d - first_date).days == o - day0 for d, o in zip(dates_check, offs_check)), "day mapping"
N = len(frame)
last_day = int(frame.day_offset.iloc[-1])
qstart = day0 + (dt.date.fromisoformat(args.query_start) - first_date).days
print(f"rows {N} first {first_date} days {last_day - day0 + 1} query from day {qstart}", flush=True)

# ---- model
model = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG)
model.load_state_dict(torch.load(MODEL_B, map_location="cpu", weights_only=True))
model = model.float().eval()
mods = model.rwkv_modules  # card, deck, note, preset, global
n_layers = [len(m.blocks) for m in mods]
D = model.d_model

# ---- static feature parts
cards, card_idx = np.unique(frame.card_id.to_numpy(), return_inverse=True)
notes, note_idx = np.unique(frame.note_id.to_numpy(), return_inverse=True)
decks, deck_idx = np.unique(frame.deck_id.to_numpy(), return_inverse=True)
presets, preset_idx = np.unique(frame.preset_id.to_numpy(), return_inverse=True)
gen = torch.Generator().manual_seed(2025)


def encodings(n, dim):
    return (torch.randint(0, ID_SPLIT, (n, dim), generator=gen).float() - (ID_SPLIT - 1) / 2)


enc = {"card": encodings(len(cards), ID_ENCODE_DIMS["card_id"]), "note": encodings(len(notes), ID_ENCODE_DIMS["note_id"]),
       "deck": encodings(len(decks), ID_ENCODE_DIMS["deck_id"]), "preset": encodings(len(presets), ID_ENCODE_DIMS["preset_id"])}


def day_enc(day_rel, first_rel):
    out = []
    for p in DAY_OFFSET_ENCODE_PERIODS:
        f = 2 * np.pi / p
        out += [np.sin(f * (day_rel % p)), np.cos(f * (day_rel % p)), np.sin(f * (first_rel % p)), np.cos(f * (first_rel % p))]
    return np.stack(out, axis=-1)


def assemble(base, ci, ni, di, pi, day_rel, first_rel):
    return torch.cat([torch.from_numpy(base.astype(np.float32)), enc["card"][ci], enc["note"][ni], enc["deck"][di],
                      enc["preset"][pi], torch.from_numpy(day_enc(day_rel, first_rel).astype(np.float32))], dim=-1)


rel = frame.day_offset.to_numpy() - day0
review_feats = assemble(frame[CARD_FEATURE_COLUMNS].to_numpy(), card_idx, note_idx, deck_idx, preset_idx,
                        rel, frame.day_offset_first.to_numpy())
assert review_feats.shape[1] == model.card_features_dim
col = {c: i for i, c in enumerate(CARD_FEATURE_COLUMNS)}

# ---- state stores: per module per layer (x_shift_time [C], S [H,K,K], x_shift_chan [C])
H = mods[0].blocks[0].time_mixer.H
K = mods[0].blocks[0].time_mixer.K


def store(n, m):
    return [(torch.zeros(n, 1, D), torch.zeros(n, 1, H, K, K), torch.zeros(n, 1, D)) for _ in range(n_layers[m])]


stores = [store(len(cards), 0), store(len(decks), 1), store(len(notes), 2), store(len(presets), 3), store(1, 4)]
seen = [np.zeros(len(cards), bool), np.zeros(len(decks), bool), np.zeros(len(notes), bool), np.zeros(len(presets), bool), np.zeros(1, bool)]


def gather(m, idx):
    st = stores[m]
    return {i: ((st[i][0][idx], st[i][1][idx]), st[i][2][idx]) for i in range(n_layers[m])}


def put(m, j, state):
    for i in range(n_layers[m]):
        (xs, s), xc = state[i]
        stores[m][i][0][j] = xs[0]
        stores[m][i][1][j] = s[0]
        stores[m][i][2][j] = xc[0]


def forward(feats, idxs, states_override=None):
    """feats [B,92]; idxs = per module index arrays (None = new/None state, B must be 1); returns (R, next states)."""
    x = model.features2card(feats)
    nexts = []
    for m in range(5):
        if states_override is not None and states_override[m] is not None:
            st = states_override[m]
        elif idxs[m] is None:
            st = None
        else:
            st = gather(m, idxs[m])
        x, nst = mods[m].forward(x, st)
        nexts.append(nst)
    xp = model.prehead_norm(x)
    p = torch.softmax(model.p_linear(model.head_p(xp)), dim=-1)
    return 1.0 - p[:, 0], nexts


# ---- tracker for query features (mirrors rwkv/run_as_rnn.RNNProcess.add_same / imm_predict)
last_row = np.full(len(cards), -1)
last_i = np.zeros(len(cards), int)
last_new = np.zeros(len(cards), int)
card_set_size = 0
today = -10 ** 9
today_reviews = 0
first_rel_card = np.zeros(len(cards))
ed_cum = frame.elapsed_days_cumulative.to_numpy(float)
es_cum = frame.elapsed_seconds_cumulative.to_numpy(float)
nan_cols = frame[["note_id_is_nan", "deck_id_is_nan", "preset_id_is_nan"]].to_numpy(float)


def query_feats(active, d, n_done, prev_day, today_now, today_count):
    r = last_row[active]
    el = (d - frame.day_offset.to_numpy()[r]).astype(float)
    base = np.zeros((len(active), len(CARD_FEATURE_COLUMNS)))
    base[:, col["scaled_elapsed_days"]] = scale_elapsed_days(el)
    base[:, col["scaled_elapsed_seconds"]] = scale_elapsed_seconds(el * 86400)
    base[:, col["elapsed_seconds_sin"]] = 0.0
    base[:, col["elapsed_seconds_cos"]] = 1.0
    cd, cs = ed_cum[r] + el, es_cum[r] + el * 86400
    base[:, col["scaled_elapsed_days_cumulative"]] = scale_elapsed_days_cumulative(cd)
    base[:, col["scaled_elapsed_seconds_cumulative"]] = scale_elapsed_seconds_cumulative(cs)
    base[:, col["elapsed_seconds_cumulative_sin"]] = np.sin((cs % 86400) * 2 * np.pi / 86400)
    base[:, col["elapsed_seconds_cumulative_cos"]] = np.cos((cs % 86400) * 2 * np.pi / 86400)
    base[:, [col["note_id_is_nan"], col["deck_id_is_nan"], col["preset_id_is_nan"]]] = nan_cols[r]
    base[:, col["day_offset_diff"]] = scale_day_offset_diff(d - prev_day)
    base[:, col["day_of_week"]] = (((d - day0) % 7) - 3) / 3
    base[:, col["diff_new_cards"]] = scale_diff_new_cards(card_set_size - last_new[active])
    base[:, col["diff_reviews"]] = scale_diff_reviews(np.maximum(0, n_done - last_i[active] - 1))
    same = d == today_now
    base[:, col["cum_reviews_today"]] = scale_cum_reviews_today(today_count if same else 0)
    base[:, col["cum_new_cards_today"]] = scale_cum_new_cards_today(0)
    base[:, col["is_query"]] = 1.0
    return assemble(base, card_idx[r], note_idx[r], deck_idx[r], preset_idx[r], np.full(len(active), d - day0),
                    first_rel_card[active])


def batched(feats, ci, overrides=None, chunk=4096):
    out = []
    r = last_row[ci]
    for s in range(0, len(ci), chunk):
        sl = slice(s, s + chunk)
        idxs = [ci[sl], deck_idx[r[sl]], note_idx[r[sl]], preset_idx[r[sl]], np.zeros(len(ci[sl]), int)]
        ov = None
        if overrides is not None:
            ov = [None if o is None else o(idxs[m]) for m, o in enumerate(overrides)]
        R, _ = forward(feats[sl], idxs, ov)
        out.append(R)
    return torch.cat(out).numpy()


days = np.arange(qstart, last_day + 1)
nd, nc = len(days), len(cards)
arr = {k: np.full((nd, nc), np.nan, np.float32) for k in ("full", "q0", "qg", "qs")}
n_rev_day = np.zeros(nd, int)

t0 = time.time()
day_rows = {}
starts = np.searchsorted(frame.day_offset.to_numpy(), np.arange(day0, last_day + 2))
n_done = 0
prev_day = None
start_d = day0
TRACK = ("n_done", "prev_day", "today", "today_reviews", "card_set_size")
CK = SCRATCH / "ckpt.pt"


def save_partial(d):
    meta = dict(days=days, first_date=str(first_date), day0=day0, n_rev_day=n_rev_day, card_ids=cards,
                card_deck=deck_idx[np.maximum(last_row, 0)], card_preset=preset_idx[np.maximum(last_row, 0)],
                card_note=note_idx[np.maximum(last_row, 0)], completed_through=d)
    np.savez_compressed(SCRATCH / "replay.npz", **arr, **meta)


def save_ckpt(d):
    torch.save({"d": d, "stores": stores, "seen": seen, "last_row": last_row, "last_i": last_i, "last_new": last_new,
                "first_rel_card": first_rel_card, "arr": arr, "n_rev_day": n_rev_day,
                **{k: globals()[k] for k in TRACK}}, CK.with_suffix(".tmp"))
    os.replace(CK.with_suffix(".tmp"), CK)
    save_partial(d)


if args.resume and CK.exists():
    ck = torch.load(CK, weights_only=False)
    stores, seen, last_row, last_i, last_new = ck["stores"], ck["seen"], ck["last_row"], ck["last_i"], ck["last_new"]
    first_rel_card, arr, n_rev_day = ck["first_rel_card"], ck["arr"], ck["n_rev_day"]
    n_done, prev_day, today, today_reviews, card_set_size = (ck[k] for k in TRACK)
    start_d = ck["d"] + 1
    print(f"resumed after day {ck['d'] - day0}", flush=True)
for d in range(start_d, last_day + 1):
    a, b = starts[d - day0], starts[d - day0 + 1]
    q = d >= qstart
    if q:
        k = d - qstart
        active = np.flatnonzero(last_row >= 0)
        if len(active):
            f0 = query_feats(active, d, n_done, prev_day, today, today_reviews + 1)
            with torch.inference_mode():
                arr["q0"][k, active] = batched(f0, active)
            snap_deck = [tuple(t.clone() for t in L) for L in stores[1]]
            snap_preset = [tuple(t.clone() for t in L) for L in stores[3]]
            snap_global = [tuple(t.clone() for t in L) for L in stores[4]]
            old_notes = {}
    # warm-up of day d
    torch.set_num_threads(1)
    with torch.inference_mode():
        for j in range(a, b):
            c, n_, dk, pr = card_idx[j], note_idx[j], deck_idx[j], preset_idx[j]
            if q and len(active) and seen[2][n_] and n_ not in old_notes:
                old_notes[n_] = [tuple(t[n_].clone() for t in L) for L in stores[2]]
            idxs = [np.array([c]) if seen[0][c] else None, np.array([dk]) if seen[1][dk] else None,
                    np.array([n_]) if seen[2][n_] else None, np.array([pr]) if seen[3][pr] else None,
                    np.array([0]) if seen[4][0] else None]
            _, nexts = forward(review_feats[j:j + 1], idxs)
            for m, jj in enumerate((c, dk, n_, pr, 0)):
                put(m, jj, nexts[m])
                seen[m][jj] = True
            # tracker
            if frame.day_offset.iat[j] != today:
                today = int(frame.day_offset.iat[j])
                today_reviews = -1
            today_reviews += 1
            if last_row[c] < 0:
                card_set_size += 1
                first_rel_card[c] = frame.day_offset_first.iat[j]
            last_row[c] = j
            last_i[c] = n_done
            last_new[c] = card_set_size
            n_done += 1
            prev_day = int(frame.day_offset.iat[j])
    torch.set_num_threads(args.threads)
    if q:
        k = d - qstart
        n_rev_day[k] = b - a
        act_now = np.flatnonzero(last_row >= 0)
        cnt = today_reviews + 1 if today == d else 0
        ff = query_feats(act_now, d, n_done, prev_day, today, cnt)
        with torch.inference_mode():
            arr["full"][k, act_now] = batched(ff, act_now)
            if len(active):
                reviewed_today = np.zeros(nc, bool)
                reviewed_today[card_idx[a:b]] = True
                nr = active[~reviewed_today[active]]
                if len(nr):
                    pos = np.searchsorted(active, nr)
                    fnr = f0[torch.from_numpy(pos)]

                    def snap(store_):
                        return lambda ix: {i: ((store_[i][0][ix], store_[i][1][ix]), store_[i][2][ix]) for i in range(len(store_))}

                    def old_note(ix):
                        out = gather(2, ix)
                        for p_, nn in enumerate(ix):
                            if nn in old_notes:
                                for i in range(n_layers[2]):
                                    (xs, s), xc = out[i]
                                    xs[p_], s[p_], xc[p_] = old_notes[nn][i][0], old_notes[nn][i][1], old_notes[nn][i][2]
                        return out
                    # q0 recomputed with exact old states is arr q0 (computed before warm-up); qg: new global only
                    arr["qg"][k, nr] = batched(fnr, nr, [None, snap(snap_deck), old_note, snap(snap_preset), None])
                    arr["qs"][k, nr] = batched(fnr, nr)
    if d == qstart - 1 or (d >= qstart and (d - qstart) % args.ckpt_days == args.ckpt_days - 1):
        save_ckpt(d)
    if (d - day0) % 20 == 0 or d == last_day:
        el = time.time() - t0
        print(f"day {d - day0}/{last_day - day0} rows {n_done}/{N} {el:.0f}s {n_done / max(el, 1e-9):.0f} rev/s", flush=True)

save_partial(last_day)
print(f"done {time.time() - t0:.0f}s", flush=True)
