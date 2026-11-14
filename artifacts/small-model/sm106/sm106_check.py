#!/usr/bin/env python
"""The Tiny two-head model (SM106): CPU proofs before any GPU run -> check-cpu-sm106.json (ALL OK required).

Design (2026-11-11, design.md): the show-time retrain (SM92)'s constant-LR epoch 26 (read only), card stream cut to
1 layer (w64-c1) -> epochs 27..29 (both heads, constant LR, show-time inputs, SM92 epoch 26's recipe) -> the release's
one-epoch 32k Muon cool-down on show-time inputs from epoch 29 (SM92's cool-down except the start and the architecture).

  K1 code: branch sm106 is cut from origin/sm101: scripts/ expertium/ rwkv/ trees of HEAD == those of 4f93a716 (the Tiny
     Instant-Only model (SM101)'s finished run) == c99330a7 (the code the Small Instant-Only model (SM98) and SM101 trained
     with); no uncommitted source. Versus SM92's own code tree (origin/sm92) the trainer differs only by the
     curve_loss_weight knob (default 1.0, multiplies the curve terms by exactly 1.0 = SM92's loss bit for bit) and the
     scorer by the 1,500 pool option; recorded as a diff list.
  K2 options: extra-options.json (the cool-down's) == SM92's (bytes up to line ends); epoch 27's options == SM92 epoch
     26's with architecture w64-c1 (no curve_loss_weight key: both heads at SM92's weights; show_time_intervals true;
     the same KD replay teacher targets).
  K3 epochs (sm106_epoch.py): epoch 27 pair (== SM92 epoch 26's step except the epoch index / start = the cut / paths /
     architecture); the generic pair rule accepts SM98's real epochs 35 vs 34 and 34 vs 33 (selfcheck); a simulated
     chain (a fake complete epoch 27 holding the cut's files) builds epoch 28 with the launcher's prepare code and passes
     the same rule; negative controls refused: architecture w64, switch removed, curve weight 0.0 added, start = SM92
     epoch 26's uncut files, epoch 26's order, update offset + 1.
  K4 the cut: sm106-cut-e26 = SM92 epoch 26 without rwkv_modules.0.blocks.1.* (19,725 values), the kept tensors and 329
     AdamW states bitwise, w64-c1 loads it strictly; parameters 331,511 -> 311,786; layer steps == w64's minus (0, 1).
     The generic cut-equivalence proofs (SM101's check-cut-equivalence.json, ALL OK) hold on this code because K1 shows
     it is the same code.
  K5 cool-down recipe: the launcher's own prepare function (PREPARE_RECIPE=1 launch-sm106.sh) prepared
     sm106-recipe-sim-dec-1ep32k-muon from a simulated complete start holding the cut (never trained); it == SM92's real
     cool-down except the start and the architecture (sm106_pair.py decay); negative controls refused: SM101's recipe
     proof (curve weight 0) as the arm, SM92's own cool-down as the arm (architecture); its start == the cut's
     master.pth; fresh Muon PASS; label windows.
  K6 rule: sm106_rule.py selftest; decide and early run end to end on known rows relabelled (temp folder): SM92 as the
     new model -> go (A curve +0.259 [.., +0.463], instant -0.101 [.., -0.015]); early SM92 - SM92 -> continue (0);
     missing rows -> exit 2.
  K7 breaks: sm106_block.py == sm101_block.py's code; daily 20:00-01:00 (Sunday from 16:00), quiet.json replaces a night.
  K8 forward pass on CPU (2 users of the 732 set with <= 6,000 reviews, show-time inputs, w8+s8, leak-free): the cut and
     SM92 epoch 26 both give finite curve and instant LogLoss (cpu-forward-smoke.jsonl, from scripts/small_model_heldout.py
     --device cpu); the cut's per-user LogLoss is within 0.01 of the uncut model's (the cut is a warm start, not a no-op).

  python sm106_check.py             -> check-cpu-sm106.json
  python sm106_check.py cooldown    -> check-cpu-sm106-cooldown.json (after the launcher prepared the real cool-down)
"""
import hashlib
import importlib.util
import json
import math
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
S92 = W / "sm92/artifacts/small-model"
E26 = S92 / "sm92-e26"
S92DEC = S92 / "sm92-e26-dec-1ep32k-muon"
S101RECIPE = W / "sm101/artifacts/small-model/sm101-recipe-sim-dec-1ep32k-muon"
E27 = SMD / "sm106-c1-e27-st"
E29 = SMD / "sm106-c1-e29-st"
CUT = SMD / "sm106-cut-e26"
RECIPE = SMD / "sm106-recipe-sim-dec-1ep32k-muon"
DEC = SMD / "sm106-c1-e29st-st-dec-1ep32k-muon"
TRACK = "E89-F65e-F77a"
BASE_COMMITS = ("4f93a716", "c99330a7")
CUT_PREFIX = "rwkv_modules.0.blocks.1."


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True, check=True).stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def k1():
    trees = {f"{c}:{d}": git("rev-parse", f"HEAD:{d}") == git("rev-parse", f"{c}:{d}")
             for c in BASE_COMMITS for d in ("scripts", "expertium", "rwkv")}
    dirty = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv")
    has_cut = git("ls-tree", "HEAD", "scripts/sm73s2.py") != "" and "w64-c1" in (WT / "expertium/variants.py").read_text(encoding="utf-8")
    vs92 = sorted(git("diff", "--name-only", "origin/sm92", "HEAD", "--", "scripts", "expertium", "rwkv").splitlines())
    model_diff = git("diff", "origin/sm92", "HEAD", "--", "expertium/model/variant_model.py")
    default_one = "self.curve_loss_weight = 1.0" in model_diff
    ok = all(trees.values()) and not dirty and has_cut and default_one
    return ok, {"trees_equal": trees, "uncommitted_source": bool(dirty), "cut_tooling_present": has_cut,
                "files_differing_from_sm92_code": vs92, "curve_loss_weight_default_1.0": default_one}


def k2():
    text = lambda p: p.read_bytes().replace(b"\r\n", b"\n")   # noqa: E731
    mine, s92 = text(HERE / "extra-options.json"), text(S92 / "sm92/extra-options.json")
    o27, o26 = (json.loads((d / "decay.json").read_text())["options"] for d in (E27, E26))
    d27 = json.loads((E27 / "decay.json").read_text())
    checks = {"cooldown_extra_eq_sm92_bytes_up_to_line_ends": mine == s92 and "curve_loss_weight" not in json.loads(mine),
              "e27_options_eq_e26_plus_arch": o27 == {**o26, "architecture": "w64-c1"} and o26.get("architecture") == "w64",
              "both_heads_no_curve_weight": "curve_loss_weight" not in o27 and "curve_loss_weight" not in o26,
              "e27_switch_arch": o27.get("show_time_intervals") is True and o27.get("architecture") == "w64-c1"
              and d27.get("architecture") == "w64-c1",
              "kd_targets_eq_sm92": o27.get("distillation") == o26.get("distillation") == json.loads(mine)["distillation"]}
    return all(checks.values()), checks


def run_epoch(*args):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm106_epoch.py"), *args], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-900:]


def k3():
    ok_pair, out_pair = run_epoch("pair", "--epoch", "27")
    ok_self, out_self = run_epoch("selfcheck")
    sys.path.insert(0, str(HERE))
    import sm106_epoch as ep
    sim27, sim28 = SMD / "sm106-c1-e27-st-simtest", SMD / "sm106-c1-e28-st-simtest"
    sim = {}
    for t in (sim27, sim28):
        shutil.rmtree(t, ignore_errors=True)
    try:
        case = sim27 / "expertium" / TRACK
        case.mkdir(parents=True)
        for f in ("protocol.json", "decay.json"):
            shutil.copy(E27 / f, sim27 / f)
        p = json.loads((sim27 / "protocol.json").read_text())
        p["full_run"]["control_dir"] = (sim27 / "control").relative_to(WT).as_posix()
        (sim27 / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
        for f in ("master.pth", "optimizer.pth"):
            shutil.copy(CUT / f, case / f)
        (case / "result.json").write_text(json.dumps({"status": "complete", "update_offset_end": 596106}), encoding="utf-8")
        protocol, decay = ep.build(28, prev=sim27, out=sim28)
        sim28.mkdir()
        ep.atomic_json(sim28 / "protocol.json", protocol)
        ep.atomic_json(sim28 / "decay.json", decay)
        fail, info = ep.pair_fail(28, d=sim28, prev=sim27, prevprev=E26)
        sim["e28_from_simulated_e27"] = (not fail, "; ".join(fail[:4]) or info[:300])
        sim["e28_offset_is_e27_end"] = protocol["full_run"]["update_offset"] == 596106
        sim["e28_arch_w64c1"] = decay["architecture"] == "w64-c1" and decay["options"]["architecture"] == "w64-c1"
        sim["e28_order_differs_from_e27"] = [r["user_id"] for r in protocol["rows"]] != [r["user_id"] for r in json.loads((E27 / "protocol.json").read_text())["rows"]]
    finally:
        for t in (sim27, sim28):
            shutil.rmtree(t, ignore_errors=True)
    config = ep.run_config()

    def opt(f):
        def edit(p, d):
            f(d["options"])
        return edit

    def arch_w64(p, d):
        d["options"]["architecture"] = "w64"
        d["architecture"] = "w64"

    def rows26(p, d):
        trims = {t["user_id"]: t["to"] for t in p.get("label_window_trims", [])}
        p["rows"] = [{"user_id": u, "train_range": [0, trims.get(u, config["review_counts"][str(u)])],
                      "available_reviews": config["review_counts"][str(u)],
                      "validation_range": [0, config["review_counts"][str(u)]]} for u in ep.order(26, config)]

    def start_uncut(p, d):
        c = E26 / "expertium" / TRACK
        p["full_run"]["init_master"], p["full_run"]["init_optimizer"] = (c / "master.pth").as_posix(), (c / "optimizer.pth").as_posix()
        d["init_master_sha256"], d["init_optimizer_sha256"] = sha(c / "master.pth"), sha(c / "optimizer.pth")

    def offset(p, d):
        p["full_run"]["update_offset"] += 1

    neg = {}
    for name, edit in (("architecture_w64", arch_w64), ("without_switch", opt(lambda o: o.pop("show_time_intervals"))),
                       ("curve_weight_0_added", opt(lambda o: o.__setitem__("curve_loss_weight", 0.0))),
                       ("epoch26_order", rows26), ("start_uncut_e26", start_uncut), ("offset_plus_1", offset)):
        tmp = SMD / "sm106-c1-e27-st-negtest"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            p = json.loads((E27 / "protocol.json").read_text())
            p["full_run"]["control_dir"] = (tmp / "control").relative_to(WT).as_posix()
            d = json.loads((E27 / "decay.json").read_text())
            edit(p, d)
            (tmp / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
            (tmp / "decay.json").write_text(json.dumps(d), encoding="utf-8")
            fail, _ = ep.pair_fail(27, d=tmp)
            neg[name] = bool(fail)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    ok = ok_pair and ok_self and all(v[0] if isinstance(v, tuple) else v for v in sim.values()) and all(neg.values())
    return ok, {"pair_e27": out_pair, "selfcheck_sm98": out_self, "simulated_chain": sim, "negative_controls_refused": neg}


def k4():
    import torch
    sys.path.insert(0, str(WT))
    from expertium.variants import build
    sys.path.insert(0, str(HERE))
    import sm106_epoch as ep
    rec = ep.cut_record()
    old = torch.load(E26 / "expertium" / TRACK / "master.pth", map_location="cpu", weights_only=True)
    new = torch.load(CUT / "master.pth", map_location="cpu", weights_only=True)
    model = build("w64-c1").float()
    model.load_state_dict(new, strict=True)
    removed = sorted(set(old) - set(new))
    kept = all(torch.equal(new[k], old[k]) for k in new)
    full = build("w64")
    eq = json.loads((SMD / "sm101/check-cut-equivalence.json").read_text())
    out = {"removed_only_card_layer_2": bool(removed) and all(k.startswith(CUT_PREFIX) for k in removed) and not (set(new) - set(old)),
           "removed_values": sum(old[k].numel() for k in removed), "kept_tensors": len(new), "kept_bitwise": kept,
           "parameters": [sum(v.numel() for v in old.values()), sum(p.numel() for p in model.parameters())],
           "adamw_moved_dropped": [rec["optimizer_states_moved"], rec["optimizer_states_dropped"]],
           "adamw_equal_after_load": rec["optimizer_states_equal_after_load"],
           "layer_steps": [len(full.layer_steps), len(model.layer_steps)],
           "steps_are_w64_minus_card1": list(model.layer_steps) == [s for s in full.layer_steps if s != (0, 1)],
           "sm101_equivalence_proofs_same_code": eq.get("ok"), "cut_master_sha256": rec["master_sha256"]}
    ok = out["removed_only_card_layer_2"] and kept and out["removed_values"] == 19725 and out["parameters"] == [331511, 311786] \
        and out["adamw_moved_dropped"] == [329, 25] and out["adamw_equal_after_load"] and out["steps_are_w64_minus_card1"] \
        and out["sm101_equivalence_proofs_same_code"] is True
    return ok, out


def dpair(arm, ref, ref_arch="w64"):
    cmd = [PY, "-X", "utf8", str(HERE / "sm106_pair.py"), "decay", "--arm", str(arm), "--ref", str(ref), "--ref-arch", ref_arch]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-400:]


def muon_lw(dec):
    muon = json.loads((HERE / "gpu" / f"check-muon-fresh-{dec.name}.json").read_text())
    lw = json.loads((HERE / "gpu" / f"check-replay-label-windows-{dec.name}.json").read_text())
    muon_ok = bool(muon.get("checks")) and all(c.get("ok") is True for c in muon["checks"].values())
    lw_ok = lw.get("ok", lw.get("status") in ("ok", "OK", "PASS")) if isinstance(lw, dict) else False
    return muon_ok, bool(lw_ok)


def k5():
    out = {"recipe_vs_sm92_cooldown": dpair(RECIPE, S92DEC)}
    out["sm101_recipe_curve_weight0_refused"] = not dpair(S101RECIPE, S92DEC)[0]
    out["sm92_cooldown_as_arm_refused"] = not dpair(S92DEC, S92DEC)[0]
    d = json.loads((RECIPE / "decay.json").read_text())
    m, l = muon_lw(RECIPE)
    out["recipe_start_is_cut"] = d.get("start_master_sha256") == sha(CUT / "master.pth")
    out["recipe_arch"] = d.get("architecture") == "w64-c1" and d["options"].get("architecture") == "w64-c1"
    out["recipe_both_heads"] = "curve_loss_weight" not in d["options"]
    out["recipe_muon_fresh"], out["recipe_label_windows"] = m, l
    return all(v[0] if isinstance(v, tuple) else v for v in out.values()), out


def cooldown_checks():
    """The real cool-down (prepared by the launcher from epoch 29): SM92's cool-down except the start and the architecture;
    the recipe proof's (w64-c1 too) except the start; its start is epoch 29's trained master.pth and update offset end;
    fresh Muon; label windows; both heads."""
    out = {"vs_sm92_cooldown": dpair(DEC, S92DEC), "vs_recipe": dpair(DEC, RECIPE, ref_arch="w64-c1")}
    d = json.loads((DEC / "decay.json").read_text())
    q = DEC / "protocol.prepared.json"
    p = json.loads((q if q.exists() else DEC / "protocol.json").read_text())
    r29 = json.loads((E29 / "expertium" / TRACK / "result.json").read_text())
    m29 = E29 / "expertium" / TRACK / "master.pth"
    out["e29_complete"] = r29.get("status") == "complete" and not r29.get("smoke_only")
    out["start_sha_is_e29"] = d.get("start_master_sha256") == sha(m29)
    out["init_master_is_e29"] = Path(str(p["full_run"].get("init_master"))).resolve() == m29.resolve()
    out["update_offset_is_e29_end"] = p["full_run"].get("update_offset") == r29.get("update_offset_end")
    out["arch_w64c1"] = d.get("architecture") == "w64-c1" and d["options"].get("architecture") == "w64-c1"
    out["both_heads"] = "curve_loss_weight" not in d["options"]
    out["muon_fresh"], out["label_windows"] = muon_lw(DEC)
    return all(v[0] if isinstance(v, tuple) else v for v in out.values()), out


def relabel(src, dst, old, new, mode="w"):
    with open(src, encoding="utf-8") as f, open(dst, mode, encoding="utf-8") as g:
        for line in f:
            if line.strip():
                x = json.loads(line)
                if str(x.get("label", "")).startswith(old):
                    x["label"] = new + x["label"][len(old):]
                    g.write(json.dumps(x) + "\n")


def k6():
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm106_rule.py"), "selftest"], capture_output=True, text=True)
    st = (r.returncode == 0, r.stdout.strip()[-700:])
    spec = importlib.util.spec_from_file_location("sm106_rule_test", HERE / "sm106_rule.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    src = S92 / "sm92/gpu/sm92-inputs.jsonl"
    l92 = "sm92-e26-dec1ep32k/w8+s8"
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        relabel(src, t / "rows.jsonl", l92, rule.LBL)
        code = rule.decide(out_dir=t, rows=t / "rows.jsonl", rows1500=t / "none.jsonl")
        res = json.loads((t / "sm106-decision.json").read_text())
        md_ok = "Verdict: **go**" in (t / "sm106-decision.md").read_text(encoding="utf-8")
        er = t / "early.jsonl"
        relabel(src, er, l92, rule.EARLY_ARM)
        relabel(src, er, l92, rule.EARLY_REF, mode="a")
        ecode = rule.early(out_dir=t, rows=er)
        eres = json.loads((t / "sm106-early.json").read_text())
        pend = rule.early(out_dir=t, rows=t / "missing.jsonl")
        pend2 = rule.decide(out_dir=t, rows=t / "missing.jsonl")
    a = res["A_new_minus_big_e2s"]
    b = res["B_report"]["new_minus_sm92_e2s"]
    dec = (code == 0 and res["verdict"] == "go" and [a["curve"]["points"], a["instant"]["points"], a["curve"]["points_ci"][1],
           a["instant"]["points_ci"][1]] == rule.SELF["sm92_A"] and b["curve"]["points"] == 0 and b["instant"]["points"] == 0 and md_ok)
    ear = ecode == 0 and eres["decision"] == "continue" and eres["curve"]["points"] == 0 and eres["instant"]["points"] == 0 \
        and pend == 2 and pend2 == 2
    info = {"selftest": st[1], "decide_on_sm92_rows": {"exit": code, "verdict": res["verdict"],
            "A": {h: [a[h]["points"], a[h]["points_ci"]] for h in ("curve", "instant")},
            "B_vs_sm92": [b["curve"]["points"], b["instant"]["points"]],
            "B_vs_sm101_instant": res["B_report"]["new_minus_sm101_e2s"]["instant"]["points"]},
            "early_on_sm92_vs_sm92_rows": {"exit": ecode, "decision": eres["decision"], "missing_rows_exit": [pend, pend2]}}
    return st[0] and dec and ear, info


def blk(now, *extra):
    return subprocess.run([PY, "-X", "utf8", str(HERE / "sm106_block.py"), "--now", now, *extra], capture_output=True, text=True).stdout.strip()


def k7():
    body = lambda p: p.read_text(encoding="utf-8").split('"""', 2)[2]   # noqa: E731
    same_code = body(HERE / "sm106_block.py") == body(SMD / "sm101/sm101_block.py")
    want = {"2026-11-12T08:00": "NEXT 720", "2026-11-12T19:59": "NEXT 1", "2026-11-12T20:30": "NOW",
            "2026-11-13T00:59": "NOW", "2026-11-13T01:00": "NEXT 1140", "2026-11-15T15:00": "NEXT 60", "2026-11-15T16:30": "NOW"}
    got = {"same_code_as_sm101_block": (same_code, same_code)}
    with tempfile.TemporaryDirectory() as t:
        empty = Path(t) / "empty-quiet.json"   # the daily breaks alone (the live quiet.json changes from day to day)
        empty.write_text("[]", encoding="utf-8")
        for now, w in want.items():
            o = blk(now, "--quiet", str(empty))
            got[now] = (o, o.startswith(w) if w == "NOW" else o == w)
        q = Path(t) / "quiet.json"
        q.write_text(json.dumps([{"start": "2026-11-12T21:00", "end": "2026-11-13T01:00", "scope": "all"},
                                 {"start": "2026-11-12T12:00", "end": "2026-11-12T13:00", "scope": "timing"}]), encoding="utf-8")
        for now, w in (("2026-11-12T12:30", "NEXT 510"), ("2026-11-12T20:30", "NEXT 30"), ("2026-11-12T21:30", "NOW")):
            o = blk(now, "--quiet", str(q))
            got[f"synthetic {now}"] = (o, o.startswith(w) if w == "NOW" else o == w)
    return all(v[1] for v in got.values()), got


def k8():
    rows = [json.loads(l) for l in (HERE / "cpu-forward-smoke.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {}
    for r in rows:
        b = r.get("benchmark") or {}
        by.setdefault(r["user_id"], {})[r["label"].split("/")[0]] = {h: b.get(h, {}).get("LogLoss") for h in ("ahead", "instant")}
    finite = all(isinstance(v, float) and math.isfinite(v) for u in by.values() for m in u.values() for v in m.values())
    pairs = {u: {h: round(m["sm106cut"][h] - m["sm92e26"][h], 5) for h in ("ahead", "instant")} for u, m in by.items()
             if "sm106cut" in m and "sm92e26" in m}
    close = bool(pairs) and all(abs(v) <= 0.01 for d in pairs.values() for v in d.values())
    ok = len(pairs) == 2 and finite and close and all(r.get("inputs", "e2s-query") == "e2s-query" for r in rows)
    return ok, {"users": sorted(by), "cut_minus_uncut_logloss": pairs, "finite": finite, "labels": sorted({r["label"] for r in rows})}


def run(checks, out_name):
    res = {}
    for name, f in checks:
        try:
            ok, info = f()
        except Exception as error:   # a failed proof, not a crash
            ok, info = False, f"{type(error).__name__}: {error}"
        res[name] = {"ok": bool(ok), "info": info}
        print(f"[sm106-check] {name} {'OK' if ok else 'FAIL'}: {json.dumps(info)[:900]}")
    allok = all(v["ok"] for v in res.values())
    res["all_ok"] = allok
    (HERE / out_name).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[sm106-check]", "ALL OK" if allok else "FAILED", f"-> {out_name}")
    return 0 if allok else 1


def main():
    if sys.argv[1:] == ["cooldown"]:
        return run((("C1_cooldown_from_e29", cooldown_checks),), "check-cpu-sm106-cooldown.json")
    return run((("K1_code", k1), ("K2_options", k2), ("K3_epochs", k3), ("K4_cut", k4), ("K5_cooldown_recipe", k5),
                ("K6_rule", k6), ("K7_breaks", k7), ("K8_cpu_forward", k8)), "check-cpu-sm106.json")


if __name__ == "__main__":
    raise SystemExit(main())
