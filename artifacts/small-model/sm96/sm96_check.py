#!/usr/bin/env python
"""The show-time instant-only small model (SM96): CPU proofs before any GPU run -> check-cpu-sm96.json (ALL OK required).

Design (user 2026-11-09 00:20): the instant-only small model (SM57) stage 4's constant-LR instant-only epoch 30 -> epoch 31
(instant-only, constant LR, show-time inputs) -> the release's one-epoch 32k Muon cool-down on show-time inputs from
epoch 31. The same shape as the show-time retrain of the release (SM92): its epoch 26 and its cool-down on show-time inputs.

  K1 code: branch sm96 is cut from sm57 035cf1dd (the code stage 4 of the instant-only small model (SM57) trained with):
     scripts/ expertium/ rwkv/ trees of HEAD == 035cf1dd's (git tree hashes, so bitwise); and sm57 contains the whole
     show-time retrain of the release (SM92) branch (merge-base == sm92 tip), its source diff to SM92 is exactly the 6
     curve-loss-weight / 1,500-pool files, and the show-time path files (gpu_r01_samples.py, sm77_warm.py,
     expertium_decay.py, gpu_r01_prefetch.py apart from its option allowlist line) are byte-identical to SM92's.
  K2 options: extra-options.json (the cool-down's) == SM57's extra-options-instant.json + {"show_time_intervals": true}
     == SM92's extra-options.json + {"curve_loss_weight": 0.0}; epoch 31's options == stage 4's epoch 30 options + the
     switch == SM92 epoch 26's options + {"curve_loss_weight": 0.0}.
  K3 epoch 31 (sm96_epoch.py pair): == stage 4's epoch-30 step (built by the same rule from epoch 29) except the input
     switch, the epoch index of the order rule, the start (epoch 30's files, hashes, update offset) and paths; plus a
     negative control (the folder without the switch, or with another option changed, is refused).
  K4 cool-down recipe: the launcher prepares the cool-down after epoch 31 with the same command that prepared the recipe
     folder sm96-inst-e30-showtime-dec-1ep32k-muon from epoch 30 (never trained); that folder == stage 4's cool-down
     except the switch only (sm57_pair.py decay --same-start), == the release's except both flags and the start, == SM92's
     except curve_loss_weight and the start; fresh Muon PASS; label windows checked. After the cool-down is prepared from
     epoch 31 the launcher runs `sm96_check.py cooldown` (-> check-cpu-sm96-cooldown.json): the same three pairs on the
     real folder (stage 4's except the start checkpoint, the switch and paths), its start == epoch 31's master.pth
     (sha256) and update offset end, fresh Muon, label windows.
  K5 rule: sm96_rule.py selftest reproduces the show-time report's stage 4 - SM92 pair.
  K6 breaks: sm96_block.py holds everything until 2026-11-09 01:00, keeps the daily 20:00-01:00 after it, and lets a
     quiet.json window replace that night's default (real quiet.json without the hold at 2026-11-08T20:02 -> NEXT 58;
     a synthetic quiet.json moving 11-10's break to 21:00).

  python sm96_check.py             -> check-cpu-sm96.json
  python sm96_check.py cooldown    -> check-cpu-sm96-cooldown.json (after the launcher prepared the real cool-down)
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
PY = sys.executable
W = Path("C:/Users/jonat/workspace/wt")
SMD = WT / "artifacts/small-model"
S57 = W / "sm57/artifacts/small-model"
RECIPE = SMD / "sm96-inst-e30-showtime-dec-1ep32k-muon"     # the cool-down recipe proof, prepared from epoch 30 (never trained)
E31 = SMD / "sm96-inst-e31-st"                              # epoch 31 (show-time inputs)
DEC = SMD / "sm96-inst-e31st-st-dec-1ep32k-muon"            # the real cool-down, prepared by the launcher after epoch 31
S4DEC = S57 / "sm57s4-inst-e30-dec-1ep32k-muon"
E30 = S57 / "sm57s4-inst-e30"
REL = W / "sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon"
SM92DEC = W / "sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon"
SM92E26 = W / "sm92/artifacts/small-model/sm92-e26"
TRACK = "E89-F65e-F77a"
BASE = "035cf1dd400d8350bbdd9d1f27d76874ffd94bb3"
DIFF6 = {"expertium/model/variant_model.py", "scripts/eval_pools.py", "scripts/expertium_options.py",
         "scripts/gpu_r01_prefetch.py", "scripts/gpu_r01_training.py", "scripts/small_model_heldout.py"}
SAME = ["scripts/gpu_r01_samples.py", "scripts/sm77_warm.py", "scripts/expertium_decay.py", "scripts/check_muon_fresh_decay.py",
        "scripts/check_replay_label_windows.py"]
SWITCH = {"show_time_intervals": True}
WEIGHT0 = {"curve_loss_weight": 0.0}


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True, check=True).stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    return ok, {"trees_equal_035cf1dd": {d: a == b for d, (a, b) in trees.items()}, "uncommitted_source": bool(dirty),
                "sm92_tip_is_ancestor": mb == sm92, "source_diff_vs_sm92": sorted(diff), "show_time_files_identical": same,
                "prefetch_diff_is_allowlist_only": pre_ok}


def k2():
    mine = json.loads((HERE / "extra-options.json").read_text())
    s57 = json.loads((S57 / "sm57/extra-options-instant.json").read_text())
    s92 = json.loads((W / "sm92/artifacts/small-model/sm92/extra-options.json").read_text())
    o31, o30, o92 = (json.loads((d / "decay.json").read_text())["options"] for d in (E31, E30, SM92E26))
    checks = {"cooldown_extra_eq_sm57_plus_switch": mine == {**s57, **SWITCH},
              "cooldown_extra_eq_sm92_plus_weight0": mine == {**s92, **WEIGHT0},
              "e31_options_eq_e30_plus_switch": o31 == {**o30, **SWITCH} and "show_time_intervals" not in o30,
              "e31_options_eq_sm92_e26_plus_weight0": o31 == {**o92, **WEIGHT0} and "curve_loss_weight" not in o92,
              "e30_options_weight0": o30.get("curve_loss_weight") == 0.0}
    return all(checks.values()), checks


def epoch_pair():
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm96_epoch.py"), "pair"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-900:]


def k3():
    ok, out = epoch_pair()
    # negative controls on a copy inside the worktree (the pair check needs the folder under the sm96 root)
    sys.path.insert(0, str(HERE))
    import sm96_epoch as ep
    neg = {}
    for name, edit in (("without_switch", lambda o: o.pop("show_time_intervals")),
                       ("weight_changed", lambda o: o.__setitem__("curve_loss_weight", 1.0)),
                       ("batch_chunks_changed", lambda o: o.__setitem__("batch_chunks", 8))):
        tmp = SMD / "sm96-inst-e31-st-negtest"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            p = json.loads((E31 / "protocol.json").read_text())
            p["full_run"]["control_dir"] = (tmp / "control").relative_to(WT).as_posix()
            d = json.loads((E31 / "decay.json").read_text())
            edit(d["options"])
            (tmp / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
            (tmp / "decay.json").write_text(json.dumps(d), encoding="utf-8")
            fail, _ = ep.pair_fail(tmp)
            neg[name] = bool(fail)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return ok and all(neg.values()), {"pair": out, "negative_controls_refused": neg}


def pair(arm, ref, flag, same=False):
    cmd = [PY, "-X", "utf8", str(S57 / "sm57/sm57_pair.py"), "decay", "--arm", str(arm), "--ref", str(ref), "--flag", json.dumps(flag)]
    if same:
        cmd.append("--same-start")
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-400:]


def muon_lw(dec):
    muon = json.loads((HERE / "gpu" / f"check-muon-fresh-{dec.name}.json").read_text())
    lw = json.loads((HERE / "gpu" / f"check-replay-label-windows-{dec.name}.json").read_text())
    muon_ok = bool(muon.get("checks")) and all(c.get("ok") is True for c in muon["checks"].values())
    lw_ok = lw.get("ok", lw.get("status") in ("ok", "OK", "PASS")) if isinstance(lw, dict) else False
    return muon_ok, bool(lw_ok)


def k4():
    """The cool-down recipe, proven on the folder the same prepare command wrote from epoch 30."""
    out = {"recipe_vs_stage4_cooldown_same_start": pair(RECIPE, S4DEC, SWITCH, True),
           "recipe_vs_release_cooldown": pair(RECIPE, REL, {**WEIGHT0, **SWITCH}),
           "recipe_vs_sm92_cooldown": pair(RECIPE, SM92DEC, WEIGHT0)}
    d, s4 = (json.loads((x / "decay.json").read_text()) for x in (RECIPE, S4DEC))
    m, l = muon_lw(RECIPE)
    out["recipe_start_is_e30"] = d.get("start_master_sha256") == sha(E30 / "expertium" / TRACK / "master.pth") == s4.get("start_master_sha256")
    out["recipe_muon_fresh"], out["recipe_label_windows"] = m, l
    return all(v[0] if isinstance(v, tuple) else v for v in out.values()), out


def cooldown_checks():
    """The real cool-down (prepared by the launcher from epoch 31): stage 4's cool-down except the start checkpoint, the
    switch and paths; == the release's except both flags and the start; == SM92's except the weight and the start; its start
    is epoch 31's trained master.pth and update offset end; fresh Muon; label windows."""
    out = {"vs_stage4_cooldown": pair(DEC, S4DEC, SWITCH),
           "vs_release_cooldown": pair(DEC, REL, {**WEIGHT0, **SWITCH}),
           "vs_sm92_cooldown": pair(DEC, SM92DEC, WEIGHT0)}
    d = json.loads((DEC / "decay.json").read_text())
    q = DEC / "protocol.prepared.json"
    p = json.loads((q if q.exists() else DEC / "protocol.json").read_text())
    r31 = json.loads((E31 / "expertium" / TRACK / "result.json").read_text())
    m31 = E31 / "expertium" / TRACK / "master.pth"
    s31 = sha(m31)
    out["e31_complete"] = r31.get("status") == "complete" and not r31.get("smoke_only")
    out["start_sha_is_e31"] = d.get("start_master_sha256") == s31
    out["init_master_is_e31"] = Path(str(p["full_run"].get("init_master"))).resolve() == m31.resolve()
    out["update_offset_is_e31_end"] = p["full_run"].get("update_offset") == r31.get("update_offset_end")
    out["muon_fresh"], out["label_windows"] = muon_lw(DEC)
    return all(v[0] if isinstance(v, tuple) else v for v in out.values()), out


def k5():
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm96_rule.py"), "selftest"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-400:]


def blk(now, *extra):
    return subprocess.run([PY, "-X", "utf8", str(HERE / "sm96_block.py"), "--now", now, *extra], capture_output=True, text=True).stdout.strip()


def k6():
    want = {"2026-11-08T20:02": "NOW", "2026-11-09T00:59": "NOW", "2026-11-09T01:00": "NEXT 1140", "2026-11-09T19:59": "NEXT 1",
            "2026-11-09T20:30": "NOW", "2026-11-10T01:00": "NEXT 1140", "2026-11-15T15:00": "NEXT 60"}
    got = {}
    for now, w in want.items():
        o = blk(now)
        got[now] = (o, o.startswith(w) if w == "NOW" else o == w)
    o = blk("2026-11-08T20:02", "--no-hold")   # the real quiet.json moved 11-08's break to 21:00
    got["no_hold 2026-11-08T20:02 (quiet.json 21:00)"] = (o, o == "NEXT 58")
    with tempfile.TemporaryDirectory() as t:
        q = Path(t) / "quiet.json"
        q.write_text(json.dumps([{"start": "2026-11-10T21:00", "end": "2026-11-11T01:00", "scope": "all"},
                                 {"start": "2026-11-10T12:00", "end": "2026-11-10T13:00", "scope": "timing"}]), encoding="utf-8")
        for now, w in (("2026-11-10T12:30", "NEXT 510"), ("2026-11-10T20:30", "NEXT 30"), ("2026-11-10T21:30", "NOW")):
            o = blk(now, "--quiet", str(q))
            got[f"synthetic {now}"] = (o, o.startswith(w) if w == "NOW" else o == w)
    return all(v[1] for v in got.values()), got


def run(checks, out_name):
    res = {}
    for name, f in checks:
        try:
            ok, info = f()
        except Exception as error:   # a failed proof, not a crash
            ok, info = False, f"{type(error).__name__}: {error}"
        res[name] = {"ok": bool(ok), "info": info}
        print(f"[sm96-check] {name} {'OK' if ok else 'FAIL'}: {json.dumps(info)[:700]}")
    allok = all(v["ok"] for v in res.values())
    res["all_ok"] = allok
    (HERE / out_name).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[sm96-check]", "ALL OK" if allok else "FAILED", f"-> {out_name}")
    return 0 if allok else 1


def main():
    if sys.argv[1:] == ["cooldown"]:
        return run((("C1_cooldown_from_e31", cooldown_checks),), "check-cpu-sm96-cooldown.json")
    return run((("K1_code", k1), ("K2_options", k2), ("K3_epoch31_pair", k3), ("K4_cooldown_recipe", k4),
                ("K5_rule_selftest", k5), ("K6_breaks", k6)), "check-cpu-sm96.json")


if __name__ == "__main__":
    raise SystemExit(main())
