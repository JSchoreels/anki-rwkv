#!/usr/bin/env python
"""Recalibrating "recalled and answered within S s" on your own collection (SM123): Platt per S and set, one shared
shift / scale of the time distribution (and a grade temperature) fitted across S, isotonic as an upper reference;
each fitted once on the reviews before the scored period (static) and refitted every 7 days on earlier reviews only
(rolling). Pre-registered in artifacts/small-model/sm123/design.md. CPU only, aggregate output.

Answer time = front + back (question shown -> grade pressed, capped at the preset's maximum answer seconds): "within
S s" always means answered within S s, reading the back included.

Reuses, read only, the per-review model outputs and the code of the timed-recall analysis on your own collection
(SM121): wt/sm121/artifacts/small-model/sm121/work/trunk-<key>.npz and wt/sm121/scripts/sm121_own.py (labels,
slices, baselines, ECE, day-block bootstrap, privacy scan, all unchanged).

  check   [--cap N]     CPU check on the first N rows (proofs K1-K6) -> check-cpu.json
  run     [--all]        full run -> sm123-result.{md,json}
"""
from argparse import ArgumentParser
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S121_ROOT = Path("C:/Users/jonat/workspace/wt/sm121")      # read only
sys.path.insert(0, str(S121_ROOT))
_spec = importlib.util.spec_from_file_location("sm121_own", S121_ROOT / "scripts" / "sm121_own.py")
s121 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s121)
from scripts.sm117_head import LOG_LO, cdf_at  # noqa: E402  (wt/sm121's copy, read only)
from scripts.sm117_timelib import median_log_s, atomic_json  # noqa: E402

OUT = ROOT / "artifacts/small-model/sm123"
S121_RESULT = S121_ROOT / "artifacts/small-model/sm121/sm121-result.json"
S_LIST = (2, 3, 4, 5, 6, 8, 10)
SETS = s121.SETS                         # Rec (Hard, Good, Easy), GE (Good, Easy), E (Easy); 0-based grades
SET_ORDER = ("E", "GE", "Rec")           # nested: E within GE within Rec
SET_NAMES = s121.SET_NAMES
MAIN = ("sm113", "small_pair_sm98", "sm106", "sm91s2")
ROW_SETS = ("bench/full", "bench/yomitan", "mac/full", "mac/yomitan")
METHODS = ("uncal", "platt_static", "shared_static", "iso_static", "platt_roll", "shared_roll", "iso_roll")
METHOD_NAMES = {"uncal": "uncalibrated (SM121)", "platt_static": "(a) Platt, static", "shared_static": "(b) shared transform, static",
                "iso_static": "(c) isotonic, static", "platt_roll": "(a) Platt, rolling 7 days",
                "shared_roll": "(b) shared transform, rolling 7 days", "iso_roll": "(c) isotonic, rolling 7 days"}
ROLL_DAYS, ROLL_ROWS, ROLL_MIN = 7, 30000, 5000
MIN_REL_PCT = 0.05                       # every difference at least 0.05% of the baseline's LogLoss
EPS = 1e-6
KNOT_X = np.r_[LOG_LO[0], LOG_LO[1:16]]  # 16 knots: bin 0 lower edge (0.25 s) ... ln 60 s (bin 15 = the cap)
LN_S = np.log(np.asarray(S_LIST, np.float64))
FIT_LOG = []                             # (fit rows max index / day / time, applied rows min index / day / time)


def below_normal():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


# ------------------------------------------------------------------ time CDF at arbitrary ln-time positions
def knot_values(pt):
    """[n, 16] CDF values at the 16 knots of each [n, 16] bin distribution (mass below the knot)."""
    return np.concatenate([np.zeros((len(pt), 1)), np.cumsum(pt[:, :15], 1)], 1)


def cdf_ln(C, pt, x):
    """P(t <= exp(x)) per row, linear in ln time inside each bin (as cdf_at); x [n] or [n, k]; at or above ln 60 s it
    is the mass below the cap, below 0.25 s zero."""
    x = np.asarray(x, np.float64)
    two = x.ndim == 2
    X = x if two else x[:, None]
    b = np.clip(np.searchsorted(KNOT_X, X, side="right") - 1, 0, 14)
    lo, hi = KNOT_X[b], KNOT_X[b + 1]
    frac = np.clip((X - lo) / (hi - lo), 0.0, 1.0)
    rows = np.arange(len(C))[:, None]
    v = C[rows, b] + frac * pt[rows, b]
    v = np.where(X < KNOT_X[0], 0.0, np.where(X >= KNOT_X[15], C[:, 15:16], v))
    return v if two else v[:, 0]


class Combo:
    """One model / head combination with its per-row arrays and the uncalibrated event probabilities."""

    def __init__(self, pg, pt):
        self.pg, self.pt = pg, pt
        self.C = {g: knot_values(pt[:, g]) for g in (1, 2, 3)}
        self.med = {g: median_log_s(pt[:, g]) for g in (1, 2, 3)}

    def time_cdf(self, g, idx, ln_tau=0.0, delta=0.0):
        """[len(idx), len(S_LIST)] P'(t <= S | g) under the shift / scale (identity at 0, 0)."""
        m = self.med[g][idx][:, None]
        x = m + (LN_S[None] - delta - m) / math.exp(ln_tau)
        return cdf_ln(self.C[g][idx], self.pt[idx, g], x)

    def grade_probs(self, idx, ln_T=0.0, c=(0.0, 0.0, 0.0)):
        z = np.log(np.clip(self.pg[idx], 1e-12, 1)) / math.exp(ln_T) + np.r_[0.0, np.asarray(c)][None]
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def events(self, idx, prm=None):
        """{(set, S): p} for rows idx; prm = shared-transform parameters (None = uncalibrated)."""
        if prm is None:
            P = self.pg[idx]
            F = {g: self.time_cdf(g, idx) for g in (1, 2, 3)}
        else:
            P = self.grade_probs(idx, prm["ln_T"], prm["c"])
            F = {g: self.time_cdf(g, idx, *prm["time"][g]) for g in (1, 2, 3)}
        out = {}
        for sn, gs in SETS.items():
            v = sum(P[:, g:g + 1] * F[g] for g in gs)
            for k, s in enumerate(S_LIST):
                out[(sn, s)] = v[:, k]
        return out


# ------------------------------------------------------------------ fits (only on the given past rows)
def note_fit(H, ths, fit_idx, app_idx):
    """Asserts and records that the fit rows are strictly earlier (row index, Anki day, review time) than every row the
    fit is applied to."""
    t = ths - 1
    fi, ai = int(fit_idx.max()), int(app_idx.min())
    fd, ad = int(H["day"][t[fit_idx]].max()), int(H["day"][t[app_idx]].min())
    fw, aw = int(H["when_ms"][t[fit_idx]].max()), int(H["when_ms"][t[app_idx]].min())
    assert fi < ai and fd < ad and fw < aw, f"fit rows not strictly earlier: {fi} {ai} {fd} {ad} {fw} {aw}"
    FIT_LOG.append((fi, ai, fd, ad, fw, aw))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def fit_platt(p, y, iters=50):
    """(a, b) of sigmoid(a logit(p) + b) by Newton on the LogLoss (step halving if the loss rises)."""
    x = logit(p)
    w = np.array([1.0, 0.0])

    def loss(w):
        z = w[0] * x + w[1]
        return float(np.mean(np.logaddexp(0, z) - y * z))
    cur = loss(w)
    for _ in range(iters):
        z = w[0] * x + w[1]
        q = 1 / (1 + np.exp(-z))
        r = q * (1 - q)
        g = np.array([np.mean((q - y) * x), np.mean(q - y)])
        Hm = np.array([[np.mean(r * x * x), np.mean(r * x)], [np.mean(r * x), np.mean(r)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        lam = 1.0
        while lam > 1e-4:
            nw = w - lam * step
            nl = loss(nw)
            if nl <= cur + 1e-15:
                break
            lam /= 2
        if lam <= 1e-4:
            break
        done = abs(cur - nl) < 1e-12
        w, cur = nw, nl
        if done:
            break
    return w


def apply_platt(w, p):
    return 1 / (1 + np.exp(-(w[0] * logit(p) + w[1])))


def fit_iso(p, y):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.clip(p, 0, 1), y)


def fit_shared(cb, H, ths, idx):
    """Shared transform on the rows idx: grade temperature + offsets (4-class LogLoss), then per grade a shift / scale
    in ln time around the row's median, fitted across S_LIST (binary LogLoss of t <= S on the rows of that grade)."""
    from scipy.optimize import minimize
    t = ths - 1
    gr, d = H["grade"][t[idx]], H["d"][t[idx]]
    lp = np.log(np.clip(cb.pg[idx], 1e-12, 1))

    def nll_grade(th):
        z = lp / math.exp(th[0]) + np.r_[0.0, th[1:]][None]
        z = z - z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), gr]))
    rg = minimize(nll_grade, np.zeros(4), method="L-BFGS-B", bounds=[(-3, 3)] + [(-10, 10)] * 3)
    prm = {"ln_T": float(rg.x[0]), "c": [float(v) for v in rg.x[1:]], "time": {}}
    for g in (1, 2, 3):
        sel = idx[gr == g]
        if len(sel) < 200:
            prm["time"][g] = (0.0, 0.0)
            continue
        y = (H["d"][t[sel]][:, None] <= 1000 * np.asarray(S_LIST)[None]).astype(np.float64)

        def nll_time(th, sel=sel, y=y, g=g):
            q = np.clip(cb.time_cdf(g, sel, th[0], th[1]), EPS, 1 - EPS)
            return float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))
        rt = minimize(nll_time, np.zeros(2), method="L-BFGS-B", bounds=[(-2, 2), (-3, 3)])
        prm["time"][g] = (float(rt.x[0]), float(rt.x[1]))
    return prm


def repair(E):
    """Running maximum over increasing S per set, then the nesting E <= GE <= Rec; returns (repaired, changed share,
    mean abs change)."""
    out = {k: v.copy() for k, v in E.items()}
    for sn in SET_ORDER:
        run = None
        for s in S_LIST:
            run = out[(sn, s)] if run is None else np.maximum(run, out[(sn, s)])
            out[(sn, s)] = run
    for lo, hi in (("E", "GE"), ("GE", "Rec")):
        for s in S_LIST:
            out[(hi, s)] = np.maximum(out[(hi, s)], out[(lo, s)])
    n = sum(len(v) for v in E.values())
    ch = sum(int((out[k] != E[k]).sum()) for k in E)
    mad = sum(float(np.abs(out[k] - E[k]).sum()) for k in E)
    return out, ch / n, mad / n


def fit_apply(cb, H, ths, U, base, fit_idx, app_idx):
    """All three methods (and the Platt-recalibrated strongest baselines) fitted on fit_idx, applied to app_idx."""
    note_fit(H, ths, fit_idx, app_idx)
    t = ths - 1
    gr, d = H["grade"][t], H["d"][t]
    ev = {}
    for sn, gs in SETS.items():
        yall = np.isin(gr, list(gs))
        for s in S_LIST:
            ev[(sn, s)] = (yall & (d <= 1000 * s)).astype(np.float64)
    pl, iso, bl, prm_pl = {}, {}, {}, {}
    for k in U:
        y = ev[k][fit_idx]
        w = fit_platt(U[k][fit_idx], y)
        prm_pl[k] = w
        pl[k] = apply_platt(w, U[k][app_idx])
        iso[k] = fit_iso(U[k][fit_idx], y).predict(np.clip(U[k][app_idx], 0, 1))
        for bk, arr in base.get(k, {}).items():
            bl.setdefault(k, {})[bk] = apply_platt(fit_platt(arr[fit_idx], y), arr[app_idx])
    prm = fit_shared(cb, H, ths, fit_idx)
    sh = cb.events(app_idx, prm)
    pl, pl_ch, pl_mad = repair(pl)
    iso, iso_ch, iso_mad = repair(iso)
    return {"platt": pl, "shared": sh, "iso": iso, "base": bl, "prm_shared": prm, "prm_platt": prm_pl,
            "repair": {"platt": (pl_ch, pl_mad), "iso": (iso_ch, iso_mad)}}


# ------------------------------------------------------------------ windows
def windows(H, ths, Sl, cap_n=None):
    """Static fit / apply index pairs for the bench and mac row sets, and the rolling blocks."""
    t = ths - 1
    n = len(ths) if cap_n is None else cap_n
    day = H["day"][t[:n]]
    assert (np.diff(ths[:n]) > 0).all() and (np.diff(H["when_ms"][t[:n]]) >= 0).all() and (np.diff(day) >= 0).all(), \
        "rows are not in time order"
    out = {}
    fb = int(np.flatnonzero(Sl["bench"][:n])[0])
    cal = np.flatnonzero(day[:fb] < day[fb])
    out["bench"] = (cal, np.arange(fb, n))
    mac = np.flatnonzero(Sl["mac"][:n])
    if len(mac):
        fm = int(mac[0])
        elig = np.flatnonzero(((H["ed"][t[:fm]] >= 1) & (H["d"][t[:fm]] > 0)) & (day[:fm] < day[fm]))
        assert len(elig) >= len(np.flatnonzero(Sl["mac"])), "not enough eligible rows before the Mac slice"
        out["mac"] = (elig[-int(Sl["mac"].sum()):], np.arange(fm, n))
    blocks = []
    d0 = int(day[fb])
    b = (day[fb:] - d0) // ROLL_DAYS
    for k in np.unique(b):
        app = fb + np.flatnonzero(b == k)
        past = np.flatnonzero(day < d0 + ROLL_DAYS * int(k))
        if len(past) < ROLL_MIN:
            continue
        blocks.append((past[-ROLL_ROWS:], app))
    out["roll"] = blocks
    return out


def recalibrate(cb, H, ths, U, base, W, log=print):
    """{method: {(set, S): p over all rows}} (NaN where not applied), plus fit summaries."""
    n = len(ths)
    nan = lambda: {k: np.full(n, np.nan) for k in U}  # noqa: E731
    R = {m: nan() for m in METHODS if m != "uncal"}
    R["uncal"] = U
    Rmac = {m: nan() for m in ("platt_static", "shared_static", "iso_static")}
    B = {"static_bench": {}, "static_mac": {}, "roll": {}}
    info = {"static": {}, "roll": {"blocks": 0, "fit_rows": [], "ln_T": [], "repair_platt": [], "repair_iso": []}}

    def put(dst, res, app, tag, bdst):
        for m, src in (("platt", res["platt"]), ("shared", res["shared"]), ("iso", res["iso"])):
            for k, v in src.items():
                dst[f"{m}_{tag}"][k][app] = v
        for k, d in res["base"].items():
            for bk, v in d.items():
                bdst.setdefault(k, {}).setdefault(bk, np.full(n, np.nan))[app] = v
    for name, dst, bdst in (("bench", R, B["static_bench"]), ("mac", Rmac, B["static_mac"])):
        if name not in W:
            continue
        fit_idx, app = W[name]
        res = fit_apply(cb, H, ths, U, base, fit_idx, app)
        put(dst, res, app, "static", bdst)
        info["static"][name] = {"fit_rows": int(len(fit_idx)), "applied_rows": int(len(app)),
                                "shared": res["prm_shared"] | {"time": {str(g): v for g, v in res["prm_shared"]["time"].items()}},
                                "platt_GE_4": [round(float(x), 4) for x in res["prm_platt"][("GE", 4)]],
                                "repair_share": {m: round(v[0], 5) for m, v in res["repair"].items()},
                                "repair_mean_abs": {m: float(f"{v[1]:.3g}") for m, v in res["repair"].items()}}
    st = time.time()
    for i, (fit_idx, app) in enumerate(W["roll"]):
        res = fit_apply(cb, H, ths, U, base, fit_idx, app)
        put(R, res, app, "roll", B["roll"])
        info["roll"]["blocks"] += 1
        info["roll"]["fit_rows"].append(len(fit_idx))
        info["roll"]["ln_T"].append(res["prm_shared"]["ln_T"])
        info["roll"]["repair_platt"].append(res["repair"]["platt"][0])
        info["roll"]["repair_iso"].append(res["repair"]["iso"][0])
        if i % 20 == 0:
            log(f"  rolling block {i + 1}/{len(W['roll'])}, {time.time() - st:.0f} s")
    r = info["roll"]
    info["roll"] = {"blocks": r["blocks"], "fit_rows_min": int(min(r["fit_rows"])), "fit_rows_max": int(max(r["fit_rows"])),
                    "temperature_range": [round(math.exp(min(r["ln_T"])), 4), round(math.exp(max(r["ln_T"])), 4)],
                    "repair_share_mean": {"platt": round(float(np.mean(r["repair_platt"])), 5),
                                          "iso": round(float(np.mean(r["repair_iso"])), 5)}}
    return R, Rmac, B, info


# ------------------------------------------------------------------ data and evaluation
def load(only):
    H = s121.history("main")
    data, ths, missing = s121.combos_available(only=only)
    if missing:
        raise SystemExit(f"missing trunk files for {missing}")
    return H, data, ths


def baseline_groups(H, ths, pg, pt, scope, cache):
    """SM121's groups(): {(set, S): (P of the bar baselines, y, valid)} for the scope's Mac window."""
    pri, cap, B, Cc, Wm = cache["pri"], cache["cap"], cache["B"], cache["Cc"], cache["Wm"][scope]
    out = {}
    for gk, P, y, valid in s121.groups(pg, pt, H, ths, B, Cc, Wm, cap):
        if gk[0] == "fast" and gk[2] in S_LIST:
            out[(gk[1], gk[2])] = ({k: P[k] for k in ("head",) + s121.FAST_BAR}, y, valid)
    return out


def strongest_names(ck):
    """SM121's strongest bar baseline per row set and event for combination ck (bench/full etc.)."""
    r = json.loads(S121_RESULT.read_text())["combos"][ck]
    return {rs: {(sn, s): r[rs][f"fast/{sn}/{s}"]["strongest_baseline"] for sn in SETS for s in S_LIST} for rs in ROW_SETS}


def summarize(g, top, bar):
    h = g["predictors"][top]
    st = g.get("strongest_baseline")
    out = {"rows": g["rows"], "event_rate": g.get("event_rate"), "logloss": h["logloss"], "ece": h.get("ece"), "auc": h.get("auc"),
           "brier": h["brier"], "mean_p": h["mean_p"], "max_bin_gap": h.get("max_bin_gap"), "strongest": st,
           "strongest_logloss": g["predictors"][st]["logloss"] if st else None}
    if st:
        v = g["vs"][st]
        out["vs_strongest"] = {"pct": round(100 * v["delta"] / g["predictors"][st]["logloss"], 3), "ci_day": v["ci_day"]}
    if "uncal" in g["vs"]:
        v = g["vs"]["uncal"]
        out["vs_uncal"] = {"pct": round(100 * v["delta"] / g["predictors"]["uncal"]["logloss"], 3), "ci_day": v["ci_day"],
                           "uncal_logloss": g["predictors"]["uncal"]["logloss"]}
    if "base_recal" in g["vs"]:
        v = g["vs"]["base_recal"]
        out["vs_strongest_recalibrated"] = {"pct": round(100 * v["delta"] / g["predictors"]["base_recal"]["logloss"], 3),
                                            "ci_day": v["ci_day"], "logloss": g["predictors"]["base_recal"]["logloss"],
                                            "ece": g["predictors"]["base_recal"].get("ece")}
    names = [pn for pn in bar if pn in g["vs"]]
    ece_ok = h.get("ece") is not None and h["ece"] < 0.01
    ci_ok = all(g["vs"][pn]["ci_day"][1] < 0 for pn in names)
    pct_ok = all(100 * g["vs"][pn]["delta"] / g["predictors"][pn]["logloss"] <= -MIN_REL_PCT for pn in names)
    out["worst_ci_upper"] = max(g["vs"][pn]["ci_day"][1] for pn in names) if names else None
    out["ece_ok"], out["beats_all"] = bool(ece_ok), bool(ci_ok and pct_ok)
    out["reliable"] = bool(ece_ok and ci_ok and pct_ok)
    out["reliable_card_block"] = bool(ece_ok and pct_ok and all(g["vs"][pn]["ci_card"][1] < 0 for pn in names))
    return out


def evaluate_combo(ck, cb, H, ths, Sl, W, cache, log=print, row_sets=ROW_SETS, calib_tables=False):
    t = ths - 1
    strong = strongest_names(ck)
    pg, pt = cb.pg, cb.pt
    base_all = {sc: baseline_groups(H, ths, pg, pt, sc, cache) for sc in ("full", "yomitan")}
    U = cb.events(np.arange(len(ths)))
    for k in U:   # K1 (always): the uncalibrated head equals SM121's groups() head
        dev = float(np.abs(U[k] - base_all["full"][k][0]["head"]).max())
        assert dev < 1e-9, f"uncalibrated head differs from SM121 at {k}: {dev}"
    # the strongest baseline arrays to recalibrate (per scope; the names come from SM121's own read-out)
    base = {}
    for rs in row_sets:
        sc = rs.split("/")[1]
        for k, nm in strong[rs].items():
            base.setdefault(k, {})[f"{sc}:{nm}"] = base_all[sc][k][0][nm]
    st = time.time()
    R, Rmac, B, info = recalibrate(cb, H, ths, U, base, W, log)
    info["fit_seconds"] = round(time.time() - st, 1)
    res = {}
    for rs in row_sets:
        sl, sc = rs.split("/")
        m = Sl[sl] & (H["yomitan"][t] if sc == "yomitan" else True)
        bd, bc = s121.block_codes(H, ths, m)
        Wd, Wc = s121.boot_weights(bd[m]), s121.boot_weights(bc[m])

        def gen():
            for k, (P0, y, valid) in base_all[sc].items():
                bars = {pn: P0[pn] for pn in s121.FAST_BAR}
                nm = f"{sc}:{strong[rs][k]}"
                for meth in METHODS:
                    src = Rmac if (sl == "mac" and meth.endswith("_static")) else R
                    P = dict(bars, head=src[meth][k])
                    if meth != "uncal":
                        P["uncal"] = U[k]
                        bsrc = B["roll"] if meth.endswith("_roll") else B["static_mac" if sl == "mac" else "static_bench"]
                        if meth.startswith("platt") and nm in bsrc.get(k, {}):
                            P["base_recal"] = bsrc[k][nm]
                    v = valid & m & np.isfinite(P["head"])
                    assert v.sum() == (valid & m).sum(), f"{ck} {rs} {meth} {k}: recalibrated values missing on scored rows"
                    yield ("fast", f"{k[0]}|{meth}", k[1]), P, y, v
        ev = s121.evaluate(gen(), m, bd, bc, Wd, Wc, bar={"fast": s121.FAST_BAR}, calibrate=False)
        out = {}
        for key, g in ev.items():
            _, sm, s = key.split("/")
            sn, meth = sm.split("|")
            out.setdefault(meth, {})[f"{sn}/{s}"] = summarize(g, "head", s121.FAST_BAR)
            if calib_tables and sn == "GE" and s == "4":
                out[meth][f"{sn}/{s}"]["reliability_equal_mass"] = g["predictors"]["head"]["reliability_equal_mass"]
        res[rs] = out
        log(f"[sm123] {ck} {rs}: {int(m.sum()):,} rows, evaluated")
    return res, info, R


# ------------------------------------------------------------------ check
def check(args):
    below_normal()
    t0, c0 = time.time(), time.process_time()
    checks = {}
    s121.BOOT = 200
    H, data, ths = load(["sm113", "small_pair_sm98", "sm106", "sm91s2"])
    t = ths - 1
    Sl = s121.slices(H, ths)
    # K1: vectorised CDF at the identity equals cdf_at
    cb = Combo(*data["sm113"])
    idx = np.arange(min(args.cap, len(ths)))
    err = max(float(np.abs(cb.time_cdf(g, idx)[:, k] - cdf_at(cb.pt[idx, g], s)).max())
              for g in (1, 2, 3) for k, s in enumerate(S_LIST))
    checks["K1_cdf_identity"] = {"pass": err < 1e-9, "max_abs": err}
    # K5: synthetic Platt, and (b) at the identity reproduces the uncalibrated probabilities
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 1000000)
    w = fit_platt(p, (rng.uniform(size=p.size) < p).astype(float))
    ident = {"ln_T": 0.0, "c": [0.0, 0.0, 0.0], "time": {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0)}}
    E0, E1 = cb.events(idx), cb.events(idx, ident)
    dev = max(float(np.abs(E0[k] - E1[k]).max()) for k in E0)
    # (b) at the identity: equal up to the float32 normalisation of the stored P(grade) (row sums off by up to 2e-7)
    checks["K5_identities"] = {"pass": bool(abs(w[0] - 1) < 0.02 and abs(w[1]) < 0.02 and dev < 1e-6),
                               "platt_on_calibrated": [round(float(x), 4) for x in w], "shared_identity_max_abs": dev}
    # the capped run: first --cap rows, one combination (SM113), all methods, 200 bootstraps
    n = min(args.cap, len(ths))
    W = windows(H, ths, Sl, cap_n=n)
    cache = baseline_cache(H, ths)
    Slc = {k: v & (np.arange(len(ths)) < n) for k, v in Sl.items()}
    FIT_LOG.clear()
    res, info, R = evaluate_combo("sm113", cb, H, ths, Slc, W, cache, row_sets=("bench/full", "bench/yomitan"))
    # K2: finite, in [0, 1] on every applied row
    app = np.zeros(len(ths), bool)
    app[W["bench"][1]] = True
    bad = 0
    for meth, E in R.items():
        for k, v in E.items():
            x = v[app]
            bad += int((~np.isfinite(x)).sum() + (x < 0).sum() + (x > 1).sum())
    checks["K2_probabilities"] = {"pass": bad == 0, "bad_values": bad, "applied_rows": int(app.sum())}
    # K3: non-decreasing in S and nested across sets (all methods after repair; shared before any repair)
    viol = {}
    for meth, E in R.items():
        v = 0
        for sn in SETS:
            for a, b in zip(S_LIST[:-1], S_LIST[1:]):
                v += int((E[(sn, b)][app] < E[(sn, a)][app] - 1e-12).sum())
        for lo, hi in (("E", "GE"), ("GE", "Rec")):
            for s in S_LIST:
                v += int((E[(hi, s)][app] < E[(lo, s)][app] - 1e-12).sum())
        viol[meth] = v
    checks["K3_monotone_nested"] = {"pass": all(v == 0 for v in viol.values()), "violations": viol,
                                    "repair_share_static": info["static"]["bench"]["repair_share"],
                                    "repair_share_roll_mean": info["roll"]["repair_share_mean"]}
    # K4: every fit strictly earlier (asserted in note_fit; recorded here)
    checks["K4_past_only"] = {"pass": len(FIT_LOG) > 0 and all(a < b and c < d and e < f for a, b, c, d, e, f in FIT_LOG),
                              "fits": len(FIT_LOG), "rolling_blocks": info["roll"]["blocks"], "static_fit_rows": info["static"]["bench"]["fit_rows"]}
    # K6: uncalibrated head on the full bench/full rows equals sm121-result.json (Good or Easy, 4 s)
    ref = json.loads(S121_RESULT.read_text())["combos"]
    k6 = {}
    m = Sl["bench"]
    for ck in MAIN:
        c = Combo(*data[ck])
        p = c.events(np.flatnonzero(m))[("GE", 4)]
        y = (np.isin(H["grade"][t][m], [2, 3]) & (H["d"][t][m] <= 4000)).astype(float)
        ll = float(s121.ll_rows(p, y).mean())
        ece = s121.calib(np.clip(p, EPS, 1 - EPS), y)["ece"]
        r = ref[ck]["bench/full"]["fast/GE/4"]["predictors"]["head"]
        k6[ck] = {"logloss": round(ll, 6), "ref_logloss": r["logloss"], "ece": ece, "ref_ece": r["ece"],
                  "pass": abs(ll - r["logloss"]) < 2e-6 and abs(ece - r["ece"]) < 2e-5}
    checks["K6_matches_sm121"] = {"pass": all(v["pass"] for v in k6.values()), **k6}
    ge4 = res["bench/full"]
    out = {"cap_rows": n, "checks": checks, "all_pass": all(c["pass"] for c in checks.values()),
           "smoke_GE_4_bench_full_capped": {meth: {k: ge4[meth]["GE/4"].get(k) for k in ("rows", "logloss", "ece")} for meth in METHODS},
           "fit_seconds": info["fit_seconds"], "seconds": round(time.time() - t0, 1), "cpu_seconds": round(time.process_time() - c0, 1),
           "peak_gb": s121.peak_gb()}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "check-cpu.json", out)
    print(json.dumps({k: v["pass"] for k, v in checks.items()}), "all_pass", out["all_pass"], out["seconds"], "s", out["peak_gb"], "GB")
    s121.privacy_scan("main", [OUT / "check-cpu.json"])
    return 0 if out["all_pass"] else 1


def baseline_cache(H, ths):
    pri = json.loads(s121.PRIORS.read_text())
    scopes = {"full": np.ones(len(H["d"]), bool), "yomitan": H["yomitan"]}
    return {"pri": pri, "cap": s121.time_user_cap(H["d"]), "B": s121.recent_baselines(H, ths, pri),
            "Cc": s121.card_counts(H, ths, s121.S_GRID, strict=False),
            "Wm": {sc: s121.mac_window(H, ths, mm, s121.S_GRID, strict=False) for sc, mm in scopes.items()}}


# ------------------------------------------------------------------ full run
def run(args):
    below_normal()
    t0, c0 = time.time(), time.process_time()
    combos = list(MAIN) + ([c for c in s121.COMBOS if c not in MAIN] if args.all else [])
    H, data, ths = load(combos)
    Sl = s121.slices(H, ths)
    W = windows(H, ths, Sl)
    cache = baseline_cache(H, ths)
    res = {"design": {"S": list(S_LIST), "sets": list(SETS), "methods": METHOD_NAMES, "rolling": {"days": ROLL_DAYS, "rows": ROLL_ROWS, "min_rows": ROLL_MIN},
                      "rule": "ECE < 0.01 and LogLoss below every SM121 bar baseline, day-block 95% CI < 0, each >= 0.05%",
                      "bootstrap": s121.BOOT, "seed": 0, "ece_bins": s121.ECE_BINS, "sm121_commit": "4e819746",
                      "data": "own collection copy of 2026-10-14 (reviews up to 2026-10-04), SM121 trunk outputs, fp32 state, show-time inputs"},
           "windows": {"static_bench_fit_rows": int(len(W["bench"][0])), "static_mac_fit_rows": int(len(W["mac"][0])),
                       "rolling_blocks": len(W["roll"])},
           "rows": {rs: int((Sl[rs.split('/')[0]] & (H["yomitan"][ths - 1] if rs.endswith("yomitan") else True)).sum()) for rs in ROW_SETS},
           "combos": {}, "fits": {}, "timing": {}}
    for ck in combos:
        st, cst = time.time(), time.process_time()
        FIT_LOG.clear()
        cb = Combo(*data[ck])
        r, info, _ = evaluate_combo(ck, cb, H, ths, Sl, W, cache, calib_tables=True)
        info["fits_checked_past_only"] = len(FIT_LOG)
        res["combos"][ck], res["fits"][ck] = r, info
        res["timing"][ck] = {"seconds": round(time.time() - st, 1), "cpu_seconds": round(time.process_time() - cst, 1)}
        print(f"[sm123] {ck} done in {time.time() - st:.0f} s", flush=True)
        del cb
    res["seconds"], res["cpu_seconds"], res["peak_gb"] = round(time.time() - t0, 1), round(time.process_time() - c0, 1), s121.peak_gb()
    res["names"] = {ck: s121.COMBOS[ck][2] for ck in combos}
    atomic_json(OUT / "sm123-result.json", res)
    write_md(res)
    s121.privacy_scan("main", [OUT / "sm123-result.md", OUT / "sm123-result.json", OUT / "check-cpu.json"])
    print(f"[sm123] done in {res['seconds']:.0f} s, cpu {res['cpu_seconds']:.0f} s, peak {res['peak_gb']} GB")
    return 0


# ------------------------------------------------------------------ report
def write_md(res):
    L = []
    A = L.append
    RS = {"bench/full": "benchmark rows, whole collection", "bench/yomitan": "benchmark rows, deck subset",
          "mac/full": "Mac slice, whole collection", "mac/yomitan": "Mac slice, deck subset"}
    A("# Recalibrating timed recall on your own collection (SM123): result\n")
    A("**What answer time means.** The whole time on the card, front and back together, from the question shown to the grade pressed, "
      "capped at the preset's maximum answer seconds. Every \"within S s\" means recalled and answered within S s, reading the back included.\n")
    A(f"CPU only, scoring only, on your own collection (the copy of 2026-10-14, reviews up to 2026-10-04; aggregate numbers only), reusing the "
      f"per-review outputs of the timed-recall analysis on your own collection (SM121). Pre-registered in `design.md`. Rows: "
      + ", ".join(f"{RS[rs]} {n:,}" for rs, n in res["rows"].items()) + ". Static fits: "
      f"{res['windows']['static_bench_fit_rows']:,} rows before the benchmark rows (2024-01 to 2024-11) for the benchmark rows, "
      f"{res['windows']['static_mac_fit_rows']:,} eligible rows just before the Mac slice for the Mac slice; rolling: "
      f"{res['windows']['rolling_blocks']} refits, every 7 days, each on the latest 30,000 time-labelled rows of earlier days.\n")
    A("**Reliable after recalibration** = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline of SM121's bar with the "
      "day-block 95% CI below 0 and each difference at least 0.05%. The baselines are not recalibrated (as in SM121); a sensitivity "
      "column compares the Platt-recalibrated head with the Platt-recalibrated strongest baseline.\n")
    sets_md = {"GE": "Good or Easy", "E": "Easy", "Rec": "recalled"}
    A("The deck subset is the subset of decks the Mac replay used (scored with the whole collection's recalibration). Easy is rare "
      "(0.1% to 1% of rows), so its ECE is always tiny; where Easy fails, it is because a baseline is as good, not calibration. Hard "
      "has only 112 rows in the static benchmark window, below the 200-row minimum, so the static shared transform keeps Hard's time "
      "distribution unchanged there.\n")
    # summary: reliable S per method / set / row set
    A("## Where it becomes reliable (S values in seconds)\n")
    for ck, r in res["combos"].items():
        A(f"### {res['names'][ck]}\n")
        A("| method | " + " | ".join(f"{sets_md[sn]}, {RS[rs]}" for sn in ("GE", "E", "Rec") for rs in ("bench/full", "bench/yomitan")) + " |")
        A("|---|" + "---|" * 6)
        for meth in METHODS:
            cells = []
            for sn in ("GE", "E", "Rec"):
                for rs in ("bench/full", "bench/yomitan"):
                    ok = [str(s) for s in S_LIST if r[rs][meth][f"{sn}/{s}"]["reliable"]]
                    cells.append(", ".join(ok) if ok else "none")
            A(f"| {METHOD_NAMES[meth]} | " + " | ".join(cells) + " |")
        A("")
        A("Mac slice (secondary), Good or Easy: " + "; ".join(
            f"{METHOD_NAMES[meth]}: whole collection " + (", ".join(str(x) for x in S_LIST if r["mac/full"][meth][f"GE/{x}"]["reliable"]) or "none")
            + ", deck subset " + (", ".join(str(x) for x in S_LIST if r["mac/yomitan"][meth][f"GE/{x}"]["reliable"]) or "none")
            for meth in METHODS) + ".\n")
    # the 4 s Good or Easy table
    A("## Good or Easy within 4 s (reading the back included)\n")
    for rs in ROW_SETS:
        A(f"### {RS[rs]}\n")
        A("| combination | method | LogLoss | ECE | vs uncalibrated | strongest baseline (LogLoss) | vs strongest [day CI of the difference] | vs strongest, both Platt | AUC | reliable |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for ck, r in res["combos"].items():
            for meth in METHODS:
                g = r[rs][meth]["GE/4"]
                vu = g.get("vs_uncal")
                vs = g.get("vs_strongest")
                vb = g.get("vs_strongest_recalibrated")
                A(f"| {res['names'][ck]} | {METHOD_NAMES[meth]} | {g['logloss']:.4f} | {g['ece']:.4f} | "
                  + (f"{vu['pct']:+.2f}%" if vu else "-") + f" | {s121.FAST_NAMES[g['strongest']]} ({g['strongest_logloss']:.4f}) | "
                  + (f"{vs['pct']:+.2f}% [{vs['ci_day'][0]:+.4f}, {vs['ci_day'][1]:+.4f}]" if vs else "-") + " | "
                  + (f"{vb['pct']:+.2f}% [{vb['ci_day'][0]:+.4f}, {vb['ci_day'][1]:+.4f}]" if vb else "-")
                  + f" | {g['auc']:.3f} | {'yes' if g['reliable'] else 'no'} |")
        A("")
    # ECE per S for the main combos, benchmark rows
    A("## ECE by S, benchmark rows, whole collection\n")
    for sn in ("GE", "E", "Rec"):
        A(f"### {sets_md[sn]}\n")
        A("| combination | method | " + " | ".join(f"{s} s" for s in S_LIST) + " |")
        A("|---|---|" + "---|" * len(S_LIST))
        for ck, r in res["combos"].items():
            for meth in METHODS:
                cells = []
                for s in S_LIST:
                    g = r["bench/full"][meth][f"{sn}/{s}"]
                    cells.append(f"{g['ece']:.4f}{'*' if g['reliable'] else ''}")
                A(f"| {res['names'][ck]} | {METHOD_NAMES[meth]} | " + " | ".join(cells) + " |")
        A("")
    A("`*` = reliable after recalibration (ECE below 0.01 and every baseline beaten).\n")
    A("## Fits\n")
    for ck, f in res["fits"].items():
        sb = f["static"]["bench"]
        sh = sb["shared"]
        A(f"- {res['names'][ck]}: static shared transform on the benchmark window: grade temperature {math.exp(sh['ln_T']):.3f}, "
          f"time scale (Hard, Good, Easy) " + ", ".join(f"{math.exp(sh['time'][g][0]):.3f}" for g in ("1", "2", "3"))
          + ", shift (ln s) " + ", ".join(f"{sh['time'][g][1]:+.3f}" for g in ("1", "2", "3"))
          + f"; Platt at Good or Easy 4 s (a, b) {sb['platt_GE_4']}; repair share (rows x events changed) Platt {sb['repair_share']['platt']:.4f}, "
          f"isotonic {sb['repair_share']['iso']:.4f}; rolling: {f['roll']['blocks']} refits, grade temperature "
          f"{f['roll']['temperature_range'][0]}-{f['roll']['temperature_range'][1]}; fits checked past-only: {f['fits_checked_past_only']}.")
    A("")
    A(f"Run time {res['seconds'] / 60:.1f} min, CPU time {res['cpu_seconds'] / 60:.1f} min, peak memory {res['peak_gb']} GB.\n")
    (OUT / "sm123-result.md").write_text("\n".join(L), encoding="utf-8")


def report(args):
    """Rewrites sm123-result.md from sm123-result.json (no recomputation) and reruns the privacy scan."""
    write_md(json.loads((OUT / "sm123-result.json").read_text()))
    s121.privacy_scan("main", [OUT / "sm123-result.md", OUT / "sm123-result.json", OUT / "check-cpu.json"])
    return 0


def main():
    p0 = ArgumentParser()
    sub = p0.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--cap", type=int, default=60000)
    sub.add_parser("report")
    p = sub.add_parser("run")
    p.add_argument("--all", action="store_true")
    a = p0.parse_args()
    return {"check": check, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
