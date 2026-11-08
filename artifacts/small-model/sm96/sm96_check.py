#!/usr/bin/env python
"""The show-time instant-only small model (SM96): CPU proofs before any GPU run -> check-cpu-sm96.json (ALL OK required).

  K1 code: branch sm96 is cut from sm57 8b323fb1 (the code stage 4 of the instant-only small model (SM57) trained with):
     scripts/ expertium/ rwkv/ trees of HEAD == 8b323fb1's (git tree hashes, so bitwise); and sm57 contains the whole
     show-time retrain of the release (SM92) branch (merge-base == sm92 tip), its source diff to SM92 is exactly the 6
     curve-loss-weight / 1,500-pool files, and the show-time path files (gpu_r01_samples.py, sm77_warm.py,
     expertium_decay.py, gpu_r01_prefetch.py apart from its option allowlist line) are byte-identical to SM92's.
  K2 options: extra-options.json == SM57's extra-options-instant.json + {"show_time_intervals": true}
     == SM92's extra-options.json + {"curve_loss_weight": 0.0}; the cool-down's decay.json options carry both.
  K3 cool-down folder (sm57_pair.py decay): == stage 4's cool-down except the flag only (--same-start: same start
     weights, offset, users, order, seeds, schedule, KD targets); == the release's cool-down except both flags and the
     start; == SM92's cool-down except curve_loss_weight and the start.
  K4 start: the cool-down starts from SM57 stage 4's constant-LR epoch 30 master.pth (sha256 == stage 4 cool-down's
     start_master_sha256), fresh Muon (check_muon_fresh_decay.py PASS), label windows trimmed + checked.
  K5 rule: sm96_rule.py selftest reproduces the show-time report's stage 4 - SM92 pair.
  K6 breaks: sm96_block.py holds everything until 2026-11-09 01:00 and keeps the daily 20:00-01:00 after it.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
PY = sys.executable
W = Path("C:/Users/jonat/workspace/wt")
S57 = W / "sm57/artifacts/small-model"
DEC = WT / "artifacts/small-model/sm96-inst-e30-showtime-dec-1ep32k-muon"
S4DEC = S57 / "sm57s4-inst-e30-dec-1ep32k-muon"
E30 = S57 / "sm57s4-inst-e30"
REL = W / "sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon"
SM92DEC = W / "sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon"
TRACK = "E89-F65e-F77a"
BASE = "8b323fb196641f7fcd513164bfb528abbb2398e2"
DIFF6 = {"expertium/model/variant_model.py", "scripts/eval_pools.py", "scripts/expertium_options.py",
         "scripts/gpu_r01_prefetch.py", "scripts/gpu_r01_training.py", "scripts/small_model_heldout.py"}
SAME = ["scripts/gpu_r01_samples.py", "scripts/sm77_warm.py", "scripts/expertium_decay.py", "scripts/check_muon_fresh_decay.py",
        "scripts/check_replay_label_windows.py"]


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True, check=True).stdout.strip()


def k1():
    trees = {d: (git("rev-parse", f"HEAD:{d}"), git("rev-parse", f"{BASE}:{d}")) for d in ("scripts", "expertium", "rwkv")}
    dirty = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv")
    sm92 = git("rev-parse", "sm92")
    mb = git("merge-base", "HEAD", "sm92")
    diff = set(git("diff", "--name-only", "sm92", "HEAD", "--", "scripts", "expertium", "rwkv").split())
    same = {f: git("rev-parse", f"sm92:{f}") == git("rev-parse", f"HEAD:{f}") for f in SAME}
    pre = git("diff", "-U0", "sm92", "HEAD", "--", "scripts/gpu_r01_prefetch.py")
    pre_lines = [l for l in pre.splitlines() if l and l[0] in "+-" and not l.startswith(("+++", "---"))]
    pre_ok = len(pre_lines) == 2 and pre_lines[0][0] == "-" and pre_lines[0][1:].strip() == '"show_time_intervals"}' and \
        pre_lines[1][0] == "+" and pre_lines[1][1:].strip() == '"show_time_intervals", "curve_loss_weight"}'
    ok = all(a == b for a, b in trees.values()) and not dirty and mb == sm92 and diff == DIFF6 and all(same.values()) and pre_ok
    return ok, {"trees_equal_8b323fb1": {d: a == b for d, (a, b) in trees.items()}, "uncommitted_source": bool(dirty),
                "sm92_tip_is_ancestor": mb == sm92, "source_diff_vs_sm92": sorted(diff), "show_time_files_identical": same,
                "prefetch_diff_is_allowlist_only": pre_ok}


def k2():
    mine = json.loads((HERE / "extra-options.json").read_text())
    s57 = json.loads((S57 / "sm57/extra-options-instant.json").read_text())
    s92 = json.loads((W / "sm92/artifacts/small-model/sm92/extra-options.json").read_text())
    opt = json.loads((DEC / "decay.json").read_text())["options"]
    ok = mine == {**s57, "show_time_intervals": True} and mine == {**s92, "curve_loss_weight": 0.0} and \
        opt.get("show_time_intervals") is True and opt.get("curve_loss_weight") == 0.0
    return ok, {"eq_sm57_plus_flag": mine == {**s57, "show_time_intervals": True}, "eq_sm92_plus_weight0": mine == {**s92, "curve_loss_weight": 0.0},
                "decay_options_show_time_intervals": opt.get("show_time_intervals"), "decay_options_curve_loss_weight": opt.get("curve_loss_weight")}


def pair(ref, flag, same=False):
    cmd = [PY, "-X", "utf8", str(S57 / "sm57/sm57_pair.py"), "decay", "--arm", str(DEC), "--ref", str(ref), "--flag", json.dumps(flag)]
    if same:
        cmd.append("--same-start")
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-400:]


def k3():
    out = {"vs_stage4_cooldown_same_start": pair(S4DEC, {"show_time_intervals": True}, True),
           "vs_release_cooldown": pair(REL, {"curve_loss_weight": 0.0, "show_time_intervals": True}),
           "vs_sm92_cooldown": pair(SM92DEC, {"curve_loss_weight": 0.0})}
    return all(v[0] for v in out.values()), out


def k4():
    d = json.loads((DEC / "decay.json").read_text())
    s4 = json.loads((S4DEC / "decay.json").read_text())
    sha = hashlib.sha256((E30 / "expertium" / TRACK / "master.pth").read_bytes()).hexdigest()
    muon = json.loads((HERE / "gpu" / f"check-muon-fresh-{DEC.name}.json").read_text())
    lw = json.loads((HERE / "gpu" / f"check-replay-label-windows-{DEC.name}.json").read_text())
    muon_ok = bool(muon.get("checks")) and all(c.get("ok") is True for c in muon["checks"].values())
    lw_ok = lw.get("ok", lw.get("status") in ("ok", "OK", "PASS")) if isinstance(lw, dict) else False
    ok = d.get("start_master_sha256") == sha == s4.get("start_master_sha256") and muon_ok and bool(lw_ok)
    return ok, {"e30_master_sha256": sha[:16], "cooldown_start_sha": str(d.get("start_master_sha256"))[:16],
                "stage4_start_sha": str(s4.get("start_master_sha256"))[:16], "muon_fresh_pass": muon_ok, "label_windows_ok": lw_ok,
                "label_windows_keys": sorted(lw)[:12] if isinstance(lw, dict) else None}


def k5():
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm96_rule.py"), "selftest"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-400:]


def k6():
    want = {"2026-11-08T20:10": "NOW", "2026-11-08T21:30": "NOW", "2026-11-09T00:59": "NOW", "2026-11-09T01:00": "NEXT 1140",
            "2026-11-09T20:30": "NOW", "2026-11-10T01:00": "NEXT 1140"}
    got = {}
    for now, w in want.items():
        o = subprocess.run([PY, "-X", "utf8", str(HERE / "sm96_block.py"), "--now", now], capture_output=True, text=True).stdout.strip()
        got[now] = (o, o.startswith(w) if w == "NOW" else o == w)
    return all(v[1] for v in got.values()), got


def main():
    res = {}
    for name, f in (("K1_code", k1), ("K2_options", k2), ("K3_cooldown_pairs", k3), ("K4_start_muon_windows", k4),
                     ("K5_rule_selftest", k5), ("K6_breaks", k6)):
        try:
            ok, info = f()
        except Exception as error:   # a failed proof, not a crash
            ok, info = False, f"{type(error).__name__}: {error}"
        res[name] = {"ok": bool(ok), "info": info}
        print(f"[sm96-check] {name} {'OK' if ok else 'FAIL'}: {json.dumps(info)[:600]}")
    allok = all(v["ok"] for v in res.values())
    res["all_ok"] = allok
    (HERE / "check-cpu-sm96.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[sm96-check]", "ALL OK" if allok else "FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
