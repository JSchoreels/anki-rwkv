#!/usr/bin/env python
"""The show-time instant-only model, stage 2 (SM98): CPU proofs before any GPU run -> check-cpu-sm98.json (ALL OK required).

Design (2026-11-09): the show-time instant-only small model (SM96)'s constant-LR epoch 31 (read only) -> epochs 32..35
(instant-only, constant LR, show-time inputs, epoch 31's recipe with the epoch index advanced) -> the release's one-epoch
32k Muon cool-down on show-time inputs from epoch 35 (SM96's cool-down except the start).

  K1 code: branch sm98 is cut from origin/sm96: scripts/ expertium/ rwkv/ trees of HEAD == those of 1e22b62c (the commit
     SM96's launcher ran epoch 31 and the cool-down from) == 8b323fb1 (the code of the instant-only small model (SM57)
     stage 4) (git tree hashes, so bitwise); no uncommitted source.
  K2 options: extra-options.json (the cool-down's) == SM96's extra-options.json (bytes); epoch 32's options == SM96 epoch
     31's == the show-time retrain of the release (SM92) epoch 26's + {"curve_loss_weight": 0.0}.
  K3 epochs (sm98_epoch.py): epoch 32 pair (== SM96 epoch 31's step except the epoch index / start / paths; epoch 31
     itself started from SM57 stage 4's epoch 30); the same generic pair rule accepts stage 4's real epochs 30 vs 29 and
     29 vs 28 (selfcheck); a simulated chain (a fake complete epoch 32 holding epoch 31's files) builds epoch 33 with the
     launcher's prepare code and passes the same pair rule (so epochs 33..35 follow the same rule); negative controls
     (switch removed, weight changed, batch_chunks changed, epoch 31's order reused, start = stage 4's epoch 30,
     update offset off by one) are refused.
  K4 cool-down recipe: the launcher's own prepare function (PREPARE_RECIPE=1 launch-sm98.sh) prepared
     sm98-recipe-e31st-dec-1ep32k-muon from SM96's epoch 31 (never trained); it == SM96's real cool-down except paths
     (sm57_pair.py decay --same-start --flag {}), == stage 4's cool-down except the switch and the start, == the release's
     except both flags and the start, == SM92's except the weight and the start; its start == epoch 31's master.pth;
     fresh Muon PASS; label windows. After epoch 35 the launcher prepares the real cool-down with the same function and
     runs `sm98_check.py cooldown` (-> check-cpu-sm98-cooldown.json): the same pairs on the real folder (SM96's except
     the start), start sha256 == epoch 35's master.pth, update offset == epoch 35's end, fresh Muon, label windows.
  K5 rule: sm98_rule.py selftest reproduces SM96's decision (-0.214 [-0.233, -0.195] vs SM92; the 1,500 pool -0.481);
     decide on SM96's rows relabelled as the new model (in a temp folder) gives primary 0 / no go and the SM92 pair -0.214.
  K6 breaks: sm98_block.py: no hard hold; daily 20:00-01:00 (Sunday from 16:00), a quiet.json window replaces that night's
     default (real quiet.json at 2026-11-08T20:02 -> NEXT 58; a synthetic quiet.json moving 11-10's break to 21:00).

  python sm98_check.py             -> check-cpu-sm98.json
  python sm98_check.py cooldown    -> check-cpu-sm98-cooldown.json (after the launcher prepared the real cool-down)
"""
import hashlib
import importlib.util
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
S96 = W / "sm96/artifacts/small-model"
E31 = S96 / "sm96-inst-e31-st"
E32 = SMD / "sm98-inst-e32-st"
E35 = SMD / "sm98-inst-e35-st"
RECIPE = SMD / "sm98-recipe-e31st-dec-1ep32k-muon"           # the cool-down recipe proof from epoch 31 (never trained)
DEC = SMD / "sm98-inst-e35st-st-dec-1ep32k-muon"             # the real cool-down, prepared by the launcher after epoch 35
S96DEC = S96 / "sm96-inst-e31st-st-dec-1ep32k-muon"
S4DEC = S57 / "sm57s4-inst-e30-dec-1ep32k-muon"
REL = W / "sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon"
SM92DEC = W / "sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon"
SM92E26 = W / "sm92/artifacts/small-model/sm92-e26"
TRACK = "E89-F65e-F77a"
BASES = {"sm96_launch_1e22b62c": "1e22b62c", "sm57_stage4_8b323fb1": "8b323fb196641f7fcd513164bfb528abbb2398e2"}
SWITCH = {"show_time_intervals": True}
WEIGHT0 = {"curve_loss_weight": 0.0}
PAIR = S57 / "sm57/sm57_pair.py"


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True, check=True).stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def k1():
    trees = {f"{d} == {name}": git("rev-parse", f"HEAD:{d}") == git("rev-parse", f"{c}:{d}")
             for d in ("scripts", "expertium", "rwkv") for name, c in BASES.items()}
    dirty = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv")
    launched = [l for l in (S96 / "sm96/gpu/launch-sm96.log").read_text(encoding="utf-8").splitlines() if "started:" in l][-1]
    started_from = launched.split("started: ")[1].split()[0]
    ok = all(trees.values()) and not dirty and started_from == BASES["sm96_launch_1e22b62c"]
    return ok, {"trees_equal": trees, "uncommitted_source": bool(dirty), "sm96_last_launch_commit": started_from}


def k2():
    mine, s96 = (HERE / "extra-options.json").read_bytes(), (S96 / "sm96/extra-options.json").read_bytes()
    o32, o31, o92 = (json.loads((d / "decay.json").read_text())["options"] for d in (E32, E31, SM92E26))
    s96dec = json.loads((S96DEC / "decay.json").read_text())["options"]
    checks = {"cooldown_extra_eq_sm96_bytes": mine == s96,
              "e32_options_eq_e31": o32 == o31,
              "e32_options_eq_sm92_e26_plus_weight0": o32 == {**o92, **WEIGHT0} and "curve_loss_weight" not in o92,
              "e32_switch_and_weight0": o32.get("show_time_intervals") is True and o32.get("curve_loss_weight") == 0.0,
              "sm96_cooldown_options_have_switch_and_weight0": s96dec.get("show_time_intervals") is True and s96dec.get("curve_loss_weight") == 0.0}
    return all(checks.values()), checks


def run_epoch(*args):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm98_epoch.py"), *args], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-900:]


def k3():
    ok_pair, out_pair = run_epoch("pair", "--epoch", "32")
    ok_self, out_self = run_epoch("selfcheck")
    sys.path.insert(0, str(HERE))
    import sm98_epoch as ep
    # simulated chain: a fake complete epoch 32 (epoch 32's folder + epoch 31's files + an end offset) -> build epoch 33
    sim32, sim33 = SMD / "sm98-inst-e32-st-simtest", SMD / "sm98-inst-e33-st-simtest"
    sim = {}
    for t in (sim32, sim33):
        shutil.rmtree(t, ignore_errors=True)
    try:
        case = sim32 / "expertium" / TRACK
        case.mkdir(parents=True)
        for f in ("protocol.json", "decay.json"):
            shutil.copy(E32 / f, sim32 / f)
        p = json.loads((sim32 / "protocol.json").read_text())
        p["full_run"]["control_dir"] = (sim32 / "control").relative_to(WT).as_posix()
        (sim32 / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
        for f in ("master.pth", "optimizer.pth"):
            shutil.copy(E31 / "expertium" / TRACK / f, case / f)
        (case / "result.json").write_text(json.dumps({"status": "complete", "update_offset_end": 706496}), encoding="utf-8")
        protocol, decay = ep.build(33, prev=sim32, out=sim33)
        sim33.mkdir()
        ep.atomic_json(sim33 / "protocol.json", protocol)
        ep.atomic_json(sim33 / "decay.json", decay)
        fail, info = ep.pair_fail(33, d=sim33, prev=sim32, prevprev=E31)
        sim["e33_from_simulated_e32"] = (not fail, "; ".join(fail[:4]) or info[:300])
        sim["e33_offset_is_e32_end"] = protocol["full_run"]["update_offset"] == 706496
        sim["e33_order_differs_from_e32"] = [r["user_id"] for r in protocol["rows"]] != [r["user_id"] for r in json.loads((E32 / "protocol.json").read_text())["rows"]]
    finally:
        for t in (sim32, sim33):
            shutil.rmtree(t, ignore_errors=True)
    # negative controls on copies of epoch 32 (the pair check needs the folder under the sm98 root)
    config = ep.run_config()

    def opt(f):
        def edit(p, d):
            f(d["options"])
        return edit

    def rows31(p, d):
        trims = {t["user_id"]: t["to"] for t in p.get("label_window_trims", [])}
        p["rows"] = [{"user_id": u, "train_range": [0, trims.get(u, config["review_counts"][str(u)])],
                      "available_reviews": config["review_counts"][str(u)],
                      "validation_range": [0, config["review_counts"][str(u)]]} for u in ep.order(31, config)]

    def start30(p, d):
        c = S57 / "sm57s4-inst-e30/expertium" / TRACK
        p["full_run"]["init_master"], p["full_run"]["init_optimizer"] = (c / "master.pth").as_posix(), (c / "optimizer.pth").as_posix()

    def offset(p, d):
        p["full_run"]["update_offset"] += 1

    neg = {}
    for name, edit in (("without_switch", opt(lambda o: o.pop("show_time_intervals"))),
                       ("weight_changed", opt(lambda o: o.__setitem__("curve_loss_weight", 1.0))),
                       ("batch_chunks_changed", opt(lambda o: o.__setitem__("batch_chunks", 8))),
                       ("epoch31_order", rows31), ("start_stage4_e30", start30), ("offset_plus_1", offset)):
        tmp = SMD / "sm98-inst-e32-st-negtest"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            p = json.loads((E32 / "protocol.json").read_text())
            p["full_run"]["control_dir"] = (tmp / "control").relative_to(WT).as_posix()
            d = json.loads((E32 / "decay.json").read_text())
            edit(p, d)
            (tmp / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
            (tmp / "decay.json").write_text(json.dumps(d), encoding="utf-8")
            fail, _ = ep.pair_fail(32, d=tmp)
            neg[name] = bool(fail)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    ok = ok_pair and ok_self and all(v[0] if isinstance(v, tuple) else v for v in sim.values()) and all(neg.values())
    return ok, {"pair_e32": out_pair, "selfcheck_stage4": out_self, "simulated_chain": sim, "negative_controls_refused": neg}


def pair(arm, ref, flag, same=False):
    cmd = [PY, "-X", "utf8", str(PAIR), "decay", "--arm", str(arm), "--ref", str(ref), "--flag", json.dumps(flag)]
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
    """The cool-down recipe, proven on the folder the launcher's prepare function wrote from SM96's epoch 31."""
    out = {"recipe_vs_sm96_cooldown_same_start": pair(RECIPE, S96DEC, {}, True),
           "recipe_vs_stage4_cooldown": pair(RECIPE, S4DEC, SWITCH),
           "recipe_vs_release_cooldown": pair(RECIPE, REL, {**WEIGHT0, **SWITCH}),
           "recipe_vs_sm92_cooldown": pair(RECIPE, SM92DEC, WEIGHT0)}
    d, s96 = (json.loads((x / "decay.json").read_text()) for x in (RECIPE, S96DEC))
    m, l = muon_lw(RECIPE)
    out["recipe_start_is_e31"] = d.get("start_master_sha256") == sha(E31 / "expertium" / TRACK / "master.pth") == s96.get("start_master_sha256")
    out["recipe_muon_fresh"], out["recipe_label_windows"] = m, l
    return all(v[0] if isinstance(v, tuple) else v for v in out.values()), out


def cooldown_checks():
    """The real cool-down (prepared by the launcher from epoch 35): SM96's cool-down except the start; stage 4's except the
    start and the switch; the release's except both flags and the start; SM92's except the weight and the start; the
    recipe proof's except the start; its start is epoch 35's trained master.pth and update offset end; fresh Muon; label
    windows."""
    out = {"vs_sm96_cooldown": pair(DEC, S96DEC, {}),
           "vs_recipe": pair(DEC, RECIPE, {}),
           "vs_stage4_cooldown": pair(DEC, S4DEC, SWITCH),
           "vs_release_cooldown": pair(DEC, REL, {**WEIGHT0, **SWITCH}),
           "vs_sm92_cooldown": pair(DEC, SM92DEC, WEIGHT0)}
    d = json.loads((DEC / "decay.json").read_text())
    q = DEC / "protocol.prepared.json"
    p = json.loads((q if q.exists() else DEC / "protocol.json").read_text())
    r35 = json.loads((E35 / "expertium" / TRACK / "result.json").read_text())
    m35 = E35 / "expertium" / TRACK / "master.pth"
    out["e35_complete"] = r35.get("status") == "complete" and not r35.get("smoke_only")
    out["start_sha_is_e35"] = d.get("start_master_sha256") == sha(m35)
    out["init_master_is_e35"] = Path(str(p["full_run"].get("init_master"))).resolve() == m35.resolve()
    out["update_offset_is_e35_end"] = p["full_run"].get("update_offset") == r35.get("update_offset_end")
    out["muon_fresh"], out["label_windows"] = muon_lw(DEC)
    return all(v[0] if isinstance(v, tuple) else v for v in out.values()), out


def k5():
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm98_rule.py"), "selftest"], capture_output=True, text=True)
    st = (r.returncode == 0, r.stdout.strip()[-600:])
    # decide end to end on SM96's rows relabelled as the new model, in a temp folder (nothing written next to the rule)
    spec = importlib.util.spec_from_file_location("sm98_rule_test", HERE / "sm98_rule.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for src, dst in ((rule.S96ROWS, t / "rows.jsonl"), (rule.S96ROWS1500, t / "rows-1500.jsonl")):
            with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
                for line in f:
                    if line.strip():
                        x = json.loads(line)
                        if str(x.get("label", "")).startswith(rule.LBL96):
                            x["label"] = rule.LBL + x["label"][len(rule.LBL96):]
                            g.write(json.dumps(x) + "\n")
        rule.HERE, rule.ROWS, rule.ROWS1500 = t, t / "rows.jsonl", t / "rows-1500.jsonl"
        code = rule.decide()
        res = json.loads((t / "sm98-decision.json").read_text())
    pi = res["primary_new_minus_sm96_e2s"]["instant"]
    s92 = res["secondary_732"]["new_minus_sm92_e2s"]["instant"]
    pool = res["pool_1500"]
    dec = (code == 0 and pi["points"] == 0 and res["verdict"] == "no go" and s92["points"] == -0.214
           and s92["points_ci"] == [-0.233, -0.195] and pool.get("users") == 1487
           and pool["new_minus_sm96"]["instant"]["pct_of_ref"] == 0 and pool["new_minus_release"]["instant"]["pct_of_ref"] == -0.481)
    info = {"selftest": st[1], "decide_on_sm96_rows": {"exit": code, "primary": pi["points"], "verdict": res["verdict"],
            "vs_sm92": [s92["points"], s92["points_ci"]], "pool_users": pool.get("users"),
            "pool_vs_release": pool["new_minus_release"]["instant"]["pct_of_ref"], "row_check": pool.get("row_check")}}
    return st[0] and dec, info


def blk(now, *extra):
    return subprocess.run([PY, "-X", "utf8", str(HERE / "sm98_block.py"), "--now", now, *extra], capture_output=True, text=True).stdout.strip()


def k6():
    want = {"2026-11-08T20:02": "NEXT 58", "2026-11-09T00:59": "NOW", "2026-11-09T01:00": "NEXT 1140",
            "2026-11-09T14:00": "NEXT 360", "2026-11-09T19:59": "NEXT 1", "2026-11-09T20:30": "NOW",
            "2026-11-10T01:00": "NEXT 1140", "2026-11-15T15:00": "NEXT 60", "2026-11-15T16:30": "NOW"}
    got = {}
    for now, w in want.items():
        o = blk(now)
        got[now] = (o, o.startswith(w) if w == "NOW" else o == w)
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
        print(f"[sm98-check] {name} {'OK' if ok else 'FAIL'}: {json.dumps(info)[:900]}")
    allok = all(v["ok"] for v in res.values())
    res["all_ok"] = allok
    (HERE / out_name).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[sm98-check]", "ALL OK" if allok else "FAILED", f"-> {out_name}")
    return 0 if allok else 1


def main():
    if sys.argv[1:] == ["cooldown"]:
        return run((("C1_cooldown_from_e35", cooldown_checks),), "check-cpu-sm98-cooldown.json")
    return run((("K1_code", k1), ("K2_options", k2), ("K3_epochs", k3), ("K4_cooldown_recipe", k4),
                ("K5_rule", k5), ("K6_breaks", k6)), "check-cpu-sm98.json")


if __name__ == "__main__":
    raise SystemExit(main())
