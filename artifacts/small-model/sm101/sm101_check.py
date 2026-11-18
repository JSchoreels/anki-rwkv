#!/usr/bin/env python
"""The smaller instant-only model (SM101): CPU proofs before any GPU run -> check-cpu-sm101.json (ALL OK required).

Design (2026-11-10, DESIGN.md): the show-time instant-only model, stage 2 (SM98)'s constant-LR epoch 35 (read only), card
stream cut to 1 layer (w64-c1) -> epochs 36..38 (instant-only, constant LR, show-time inputs, SM98's epoch recipe) -> the
release's one-epoch 32k Muon cool-down on show-time inputs from epoch 38 (SM98's cool-down except the start and the
architecture).

  K1 code: branch sm101 is cut from origin/sm98: scripts/ expertium/ rwkv/ trees of HEAD == those of c99330a7 (the code
     SM96 and SM98 trained with; it contains the card-layer cut tooling of the card-stream-1-layer check (SM73 stage 2))
     (git tree hashes, so bitwise); no uncommitted source.
  K2 options: extra-options.json (the cool-down's) == SM98's == SM96's (bytes, up to CRLF / LF line ends); epoch 36's options == SM98 epoch 35's with
     architecture w64-c1 (== the show-time retrain of the release (SM92) epoch 26's + curve_loss_weight 0 + the arch).
  K3 epochs (sm101_epoch.py): epoch 36 pair (== SM98 epoch 35's step except the epoch index / start = the cut / paths /
     architecture); the same generic pair rule accepts SM98's real epochs 35 vs 34 and 34 vs 33 (selfcheck); a simulated
     chain (a fake complete epoch 36 holding the cut's files) builds epoch 37 with the launcher's prepare code and passes
     the same rule; negative controls refused: architecture w64, switch removed, weight 1.0, start = SM98 epoch 35's
     uncut files, epoch 35's order, update offset + 1.
  K4 the cut: sm101-cut-e35 = SM98 epoch 35 without rwkv_modules.0.blocks.1.* (19,725 values), the 390 kept tensors and
     329 AdamW states bitwise, w64-c1 loads it strictly; parameters 331,511 -> 311,786. Plus the generic cut proofs of
     the card-stream-1-layer check (SM73 stage 2) re-run on this code (scripts/sm73s2.py cpu-check ->
     check-cut-equivalence.json, ALL OK: trainer path == w64 with the layer step skipped, bitwise, eager and scripted;
     scorer path == w64 with the card layer's W_o zeroed, bitwise; Muon fresh check on a w64-c1 layout).
  K5 cool-down recipe: the launcher's own prepare function (PREPARE_RECIPE=1 launch-sm101.sh) prepared
     sm101-recipe-sim-dec-1ep32k-muon from a simulated complete start holding the cut (never trained); it == SM98's real
     cool-down except the start and the architecture (sm101_pair.py decay), == SM96's except the same; its start ==
     the cut's master.pth; fresh Muon PASS; label windows. After epoch 38 the launcher prepares the real cool-down with
     the same function and runs `sm101_check.py cooldown` (-> check-cpu-sm101-cooldown.json).
  K6 rule: sm101_rule.py selftest; early and decide run end to end on known rows relabelled (temp folder): SM98 as the
     new model -> go (A +0.110, B -0.248); early SM98 - SM96 as arm / ref -> continue (-0.035).
  K7 breaks: sm101_block.py == sm98_block.py's code; daily 20:00-01:00 (Sunday from 16:00), quiet.json replaces a night.
  K8 bytes: the fork's state layout (rslib/src/rwkv/precision.rs, ModuleState::compact; formula of the SM73 stage 2
     addendum) reproduces 4,554 / 3,018 B per card + note for w64 and gives the w64-c1 numbers alone and paired.

  python sm101_check.py             -> check-cpu-sm101.json
  python sm101_check.py cooldown    -> check-cpu-sm101-cooldown.json (after the launcher prepared the real cool-down)
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
S96 = W / "sm96/artifacts/small-model"
S98 = W / "sm98/artifacts/small-model"
E35 = S98 / "sm98-inst-e35-st"
E36 = SMD / "sm101-c1-e36-st"
E38 = SMD / "sm101-c1-e38-st"
CUT = SMD / "sm101-cut-e35"
SIMSTART = SMD / "sm101-recipe-start-sim"
RECIPE = SMD / "sm101-recipe-sim-dec-1ep32k-muon"
DEC = SMD / "sm101-c1-e38st-st-dec-1ep32k-muon"
S98DEC = S98 / "sm98-inst-e35st-st-dec-1ep32k-muon"
S96DEC = S96 / "sm96-inst-e31st-st-dec-1ep32k-muon"
SM92E26 = W / "sm92/artifacts/small-model/sm92-e26"
TRACK = "E89-F65e-F77a"
BASE_COMMIT = "c99330a7"
WEIGHT0 = {"curve_loss_weight": 0.0}
CUT_PREFIX = "rwkv_modules.0.blocks.1."


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True, check=True).stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def k1():
    trees = {d: git("rev-parse", f"HEAD:{d}") == git("rev-parse", f"{BASE_COMMIT}:{d}") for d in ("scripts", "expertium", "rwkv")}
    dirty = git("status", "--porcelain", "--", "scripts", "expertium", "rwkv")
    launched = [l for l in (S98 / "sm98/gpu/launch-sm98.log").read_text(encoding="utf-8").splitlines() if "started:" in l][-1]
    sm98_from = launched.split("started: ")[1].split()[0]
    sm98_tree = {d: subprocess.run(["git", "-C", str(W / "sm98"), "rev-parse", f"{sm98_from}:{d}"], capture_output=True, text=True).stdout.strip()
                 == git("rev-parse", f"HEAD:{d}") for d in ("scripts", "expertium", "rwkv")}
    has_cut = git("ls-tree", "HEAD", "scripts/sm73s2.py") != "" and "w64-c1" in (WT / "expertium/variants.py").read_text(encoding="utf-8")
    ok = all(trees.values()) and all(sm98_tree.values()) and not dirty and has_cut
    return ok, {"trees_equal_c99330a7": trees, "trees_equal_sm98_last_launch": [sm98_from, sm98_tree],
                "uncommitted_source": bool(dirty), "cut_tooling_present": has_cut}


def k2():
    text = lambda p: p.read_bytes().replace(b"\r\n", b"\n")   # noqa: E731  (a checkout may turn LF into CRLF)
    mine, s98, s96 = text(HERE / "extra-options.json"), text(S98 / "sm98/extra-options.json"), text(S96 / "sm96/extra-options.json")
    o36, o35, o92 = (json.loads((d / "decay.json").read_text())["options"] for d in (E36, E35, SM92E26))
    d36 = json.loads((E36 / "decay.json").read_text())
    checks = {"cooldown_extra_eq_sm98_eq_sm96_bytes_up_to_line_ends": mine == s98 == s96 and json.loads(mine) == json.loads(s98),
              "e36_options_eq_e35_plus_arch": o36 == {**o35, "architecture": "w64-c1"} and o35.get("architecture") == "w64",
              "e35_options_eq_sm92_e26_plus_weight0": o35 == {**o92, **WEIGHT0} and "curve_loss_weight" not in o92,
              "e36_switch_weight0_arch": o36.get("show_time_intervals") is True and o36.get("curve_loss_weight") == 0.0
              and o36.get("architecture") == "w64-c1" and d36.get("architecture") == "w64-c1"}
    return all(checks.values()), checks


def run_epoch(*args):
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm101_epoch.py"), *args], capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-900:]


def k3():
    ok_pair, out_pair = run_epoch("pair", "--epoch", "36")
    ok_self, out_self = run_epoch("selfcheck")
    sys.path.insert(0, str(HERE))
    import sm101_epoch as ep
    sim36, sim37 = SMD / "sm101-c1-e36-st-simtest", SMD / "sm101-c1-e37-st-simtest"
    sim = {}
    for t in (sim36, sim37):
        shutil.rmtree(t, ignore_errors=True)
    try:
        case = sim36 / "expertium" / TRACK
        case.mkdir(parents=True)
        for f in ("protocol.json", "decay.json"):
            shutil.copy(E36 / f, sim36 / f)
        p = json.loads((sim36 / "protocol.json").read_text())
        p["full_run"]["control_dir"] = (sim36 / "control").relative_to(WT).as_posix()
        (sim36 / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
        for f in ("master.pth", "optimizer.pth"):
            shutil.copy(CUT / f, case / f)
        (case / "result.json").write_text(json.dumps({"status": "complete", "update_offset_end": 794808}), encoding="utf-8")
        protocol, decay = ep.build(37, prev=sim36, out=sim37)
        sim37.mkdir()
        ep.atomic_json(sim37 / "protocol.json", protocol)
        ep.atomic_json(sim37 / "decay.json", decay)
        fail, info = ep.pair_fail(37, d=sim37, prev=sim36, prevprev=E35)
        sim["e37_from_simulated_e36"] = (not fail, "; ".join(fail[:4]) or info[:300])
        sim["e37_offset_is_e36_end"] = protocol["full_run"]["update_offset"] == 794808
        sim["e37_arch_w64c1"] = decay["architecture"] == "w64-c1" and decay["options"]["architecture"] == "w64-c1"
        sim["e37_order_differs_from_e36"] = [r["user_id"] for r in protocol["rows"]] != [r["user_id"] for r in json.loads((E36 / "protocol.json").read_text())["rows"]]
    finally:
        for t in (sim36, sim37):
            shutil.rmtree(t, ignore_errors=True)
    config = ep.run_config()

    def opt(f):
        def edit(p, d):
            f(d["options"])
        return edit

    def arch_w64(p, d):
        d["options"]["architecture"] = "w64"
        d["architecture"] = "w64"

    def rows35(p, d):
        trims = {t["user_id"]: t["to"] for t in p.get("label_window_trims", [])}
        p["rows"] = [{"user_id": u, "train_range": [0, trims.get(u, config["review_counts"][str(u)])],
                      "available_reviews": config["review_counts"][str(u)],
                      "validation_range": [0, config["review_counts"][str(u)]]} for u in ep.order(35, config)]

    def start_uncut(p, d):
        c = E35 / "expertium" / TRACK
        p["full_run"]["init_master"], p["full_run"]["init_optimizer"] = (c / "master.pth").as_posix(), (c / "optimizer.pth").as_posix()
        d["init_master_sha256"], d["init_optimizer_sha256"] = sha(c / "master.pth"), sha(c / "optimizer.pth")

    def offset(p, d):
        p["full_run"]["update_offset"] += 1

    neg = {}
    for name, edit in (("architecture_w64", arch_w64), ("without_switch", opt(lambda o: o.pop("show_time_intervals"))),
                       ("weight_changed", opt(lambda o: o.__setitem__("curve_loss_weight", 1.0))),
                       ("epoch35_order", rows35), ("start_uncut_e35", start_uncut), ("offset_plus_1", offset)):
        tmp = SMD / "sm101-c1-e36-st-negtest"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            p = json.loads((E36 / "protocol.json").read_text())
            p["full_run"]["control_dir"] = (tmp / "control").relative_to(WT).as_posix()
            d = json.loads((E36 / "decay.json").read_text())
            edit(p, d)
            (tmp / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
            (tmp / "decay.json").write_text(json.dumps(d), encoding="utf-8")
            fail, _ = ep.pair_fail(36, d=tmp)
            neg[name] = bool(fail)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    ok = ok_pair and ok_self and all(v[0] if isinstance(v, tuple) else v for v in sim.values()) and all(neg.values())
    return ok, {"pair_e36": out_pair, "selfcheck_sm98": out_self, "simulated_chain": sim, "negative_controls_refused": neg}


def k4():
    import torch
    sys.path.insert(0, str(WT))
    from expertium.variants import build
    sys.path.insert(0, str(HERE))
    import sm101_epoch as ep
    rec = ep.cut_record()
    old = torch.load(E35 / "expertium" / TRACK / "master.pth", map_location="cpu", weights_only=True)
    new = torch.load(CUT / "master.pth", map_location="cpu", weights_only=True)
    model = build("w64-c1").float()
    model.load_state_dict(new, strict=True)
    removed = sorted(set(old) - set(new))
    kept = all(torch.equal(new[k], old[k]) for k in new)
    full = build("w64")
    eq = json.loads((HERE / "check-cut-equivalence.json").read_text())
    out = {"removed_only_card_layer_2": bool(removed) and all(k.startswith(CUT_PREFIX) for k in removed) and not (set(new) - set(old)),
           "removed_values": sum(old[k].numel() for k in removed), "kept_tensors": len(new), "kept_bitwise": kept,
           "parameters": [sum(v.numel() for v in old.values()), sum(p.numel() for p in model.parameters())],
           "adamw_moved_dropped": [rec["optimizer_states_moved"], rec["optimizer_states_dropped"]],
           "adamw_equal_after_load": rec["optimizer_states_equal_after_load"],
           "layer_steps": [len(full.layer_steps), len(model.layer_steps)],
           "steps_are_w64_minus_card1": list(model.layer_steps) == [s for s in full.layer_steps if s != (0, 1)],
           "equivalence_proofs_on_this_code": {k: v["ok"] for k, v in eq["checks"].items()}, "equivalence_all_ok": eq.get("ok")}
    ok = out["removed_only_card_layer_2"] and kept and out["removed_values"] == 19725 and out["parameters"] == [331511, 311786] \
        and out["adamw_moved_dropped"] == [329, 25] and out["adamw_equal_after_load"] and out["steps_are_w64_minus_card1"] \
        and out["equivalence_all_ok"] is True
    return ok, out


def dpair(arm, ref, ref_arch="w64"):
    cmd = [PY, "-X", "utf8", str(HERE / "sm101_pair.py"), "decay", "--arm", str(arm), "--ref", str(ref), "--ref-arch", ref_arch]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stdout.strip()[-400:]


def muon_lw(dec):
    muon = json.loads((HERE / "gpu" / f"check-muon-fresh-{dec.name}.json").read_text())
    lw = json.loads((HERE / "gpu" / f"check-replay-label-windows-{dec.name}.json").read_text())
    muon_ok = bool(muon.get("checks")) and all(c.get("ok") is True for c in muon["checks"].values())
    lw_ok = lw.get("ok", lw.get("status") in ("ok", "OK", "PASS")) if isinstance(lw, dict) else False
    return muon_ok, bool(lw_ok)


def k5():
    out = {"recipe_vs_sm98_cooldown": dpair(RECIPE, S98DEC), "recipe_vs_sm96_cooldown": dpair(RECIPE, S96DEC)}
    # negative control: the uncut w64 SM98 cool-down is refused as an arm (architecture)
    out["sm98_cooldown_as_arm_refused"] = not dpair(S98DEC, S96DEC)[0]
    d = json.loads((RECIPE / "decay.json").read_text())
    m, l = muon_lw(RECIPE)
    out["recipe_start_is_cut"] = d.get("start_master_sha256") == sha(CUT / "master.pth")
    out["recipe_arch"] = d.get("architecture") == "w64-c1" and d["options"].get("architecture") == "w64-c1"
    out["recipe_muon_fresh"], out["recipe_label_windows"] = m, l
    return all(v[0] if isinstance(v, tuple) else v for v in out.values()), out


def cooldown_checks():
    """The real cool-down (prepared by the launcher from epoch 38): SM98's cool-down except the start and the architecture;
    SM96's except the same; the recipe proof's (w64-c1 too) except the start; its start is epoch 38's trained master.pth and update
    offset end; fresh Muon; label windows."""
    out = {"vs_sm98_cooldown": dpair(DEC, S98DEC), "vs_sm96_cooldown": dpair(DEC, S96DEC)}
    out["vs_recipe"] = dpair(DEC, RECIPE, ref_arch="w64-c1")
    d = json.loads((DEC / "decay.json").read_text())
    q = DEC / "protocol.prepared.json"
    p = json.loads((q if q.exists() else DEC / "protocol.json").read_text())
    r38 = json.loads((E38 / "expertium" / TRACK / "result.json").read_text())
    m38 = E38 / "expertium" / TRACK / "master.pth"
    out["e38_complete"] = r38.get("status") == "complete" and not r38.get("smoke_only")
    out["start_sha_is_e38"] = d.get("start_master_sha256") == sha(m38)
    out["init_master_is_e38"] = Path(str(p["full_run"].get("init_master"))).resolve() == m38.resolve()
    out["update_offset_is_e38_end"] = p["full_run"].get("update_offset") == r38.get("update_offset_end")
    out["arch_w64c1"] = d.get("architecture") == "w64-c1" and d["options"].get("architecture") == "w64-c1"
    out["muon_fresh"], out["label_windows"] = muon_lw(DEC)
    return all(v[0] if isinstance(v, tuple) else v for v in out.values()), out


def relabel(src, dst, old, new):
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if line.strip():
                x = json.loads(line)
                if str(x.get("label", "")).startswith(old):
                    x["label"] = new + x["label"][len(old):]
                    g.write(json.dumps(x) + "\n")


def k6():
    r = subprocess.run([PY, "-X", "utf8", str(HERE / "sm101_rule.py"), "selftest"], capture_output=True, text=True)
    st = (r.returncode == 0, r.stdout.strip()[-700:])
    spec = importlib.util.spec_from_file_location("sm101_rule_test", HERE / "sm101_rule.py")
    rule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rule)
    S98G, S96G = rule.S98G, rule.S96G
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        relabel(S98G / "sm98-rows.jsonl", t / "rows.jsonl", rule.LBL98, rule.LBL)
        relabel(S98G / "sm98-rows-1500.jsonl", t / "rows-1500.jsonl", rule.LBL98, rule.LBL)
        code = rule.decide(out_dir=t, rows=t / "rows.jsonl", rows1500=t / "rows-1500.jsonl")
        res = json.loads((t / "sm101-decision.json").read_text())
        md_ok = "Verdict: **go**" in (t / "sm101-decision.md").read_text(encoding="utf-8")
        early_rows = t / "early.jsonl"
        relabel(S98G / "sm98-rows.jsonl", early_rows, rule.LBL98, rule.EARLY_ARM)
        relabel(S96G / "sm96-rows.jsonl", t / "e96.jsonl", rule.LBL96, rule.EARLY_REF)
        with open(early_rows, "a", encoding="utf-8") as g:
            g.write((t / "e96.jsonl").read_text(encoding="utf-8"))
        ecode = rule.early(out_dir=t, rows=early_rows)
        eres = json.loads((t / "sm101-early.json").read_text())
        pend = rule.early(out_dir=t, rows=t / "missing.jsonl")
    sec = res["secondary_732"]
    dec = (code == 0 and res["verdict"] == "go" and res["A_bar_new_minus_big_normal"]["points"] == 0.11
           and res["B_new_minus_sm92_e2s"]["points"] == -0.248 and sec["new_minus_sm98_e2s"]["instant"]["points"] == 0
           and sec["new_minus_sm96_e2s"]["instant"]["points"] == -0.035 and res["pool_1500"].get("users", 0) > 1400
           and res["pool_1500"]["new_minus_sm98"]["instant"]["pct_of_ref"] == 0 and md_ok)
    ear = ecode == 0 and eres["decision"] == "continue" and eres["instant"]["points"] == -0.035 and pend == 2
    info = {"selftest": st[1], "decide_on_sm98_rows": {"exit": code, "verdict": res["verdict"],
            "A": [res["A_bar_new_minus_big_normal"]["points"], res["A_bar_new_minus_big_normal"]["points_ci"]],
            "B": [res["B_new_minus_sm92_e2s"]["points"], res["B_new_minus_sm92_e2s"]["points_ci"]],
            "vs_sm96_e2s": sec["new_minus_sm96_e2s"]["instant"]["points"], "pool_users": res["pool_1500"].get("users"),
            "row_check": res["pool_1500"].get("row_check")},
            "early_on_sm98_vs_sm96_rows": {"exit": ecode, "decision": eres["decision"], "points": eres["instant"]["points"],
                                           "missing_rows_exit": pend}}
    return st[0] and dec and ear, info


def blk(now, *extra):
    return subprocess.run([PY, "-X", "utf8", str(HERE / "sm101_block.py"), "--now", now, *extra], capture_output=True, text=True).stdout.strip()


def k7():
    body = lambda p: p.read_text(encoding="utf-8").split('"""', 2)[2]   # noqa: E731
    same_code = body(HERE / "sm101_block.py") == body(S98 / "sm98/sm98_block.py")
    want = {"2026-11-10T11:30": "NEXT 510", "2026-11-10T19:59": "NEXT 1", "2026-11-10T20:30": "NOW",
            "2026-11-11T00:59": "NOW", "2026-11-11T01:00": "NEXT 1140", "2026-11-15T15:00": "NEXT 60", "2026-11-15T16:30": "NOW"}
    got = {"same_code_as_sm98_block": (same_code, same_code)}
    for now, w in want.items():
        o = blk(now)
        got[now] = (o, o.startswith(w) if w == "NOW" else o == w)
    with tempfile.TemporaryDirectory() as t:
        q = Path(t) / "quiet.json"
        q.write_text(json.dumps([{"start": "2026-11-11T21:00", "end": "2026-11-12T01:00", "scope": "all"},
                                 {"start": "2026-11-11T12:00", "end": "2026-11-11T13:00", "scope": "timing"}]), encoding="utf-8")
        for now, w in (("2026-11-11T12:30", "NEXT 510"), ("2026-11-11T20:30", "NEXT 30"), ("2026-11-11T21:30", "NOW")):
            o = blk(now, "--quiet", str(q))
            got[f"synthetic {now}"] = (o, o.startswith(w) if w == "NOW" else o == w)
    return all(v[1] for v in got.values()), got


def state_bytes(card_layers, bits=8, paired_with=None):
    """Bytes per card + its note in the fork's layout: 14-byte magic + 1 precision byte + u32 layer count per entity; per
    layer three u32 lengths, an f16 time shift (64 x 2 B), 64 matrix rows of a 4-byte scale + 16 int8 / 8 int4 bytes, and
    an f16 channel shift (128 B) where the layer has a channel mixer (layer 0 of each stream; the card stream's layer 1 has
    none). Paired (rslib/src/rwkv/paired.rs): per entity 15-byte magic + u32 length + release state + instant state."""
    row = 4 + (16 if bits == 8 else 8)
    layer = lambda cm: 12 + 128 + 64 * row + (128 if cm else 0)   # noqa: E731
    card = 19 + sum(layer(i == 0) for i in range(card_layers))
    note = 19 + layer(True)
    if paired_with is None:
        return {"card": card, "note": note, "total": card + note}
    rel = state_bytes(paired_with, bits)
    return {"card": 19 + rel["card"] + card, "note": 19 + rel["note"] + note, "total": 38 + rel["total"] + card + note}


def k8():
    out = {f"{name}_int{bits}": state_bytes(layers, bits) for name, layers in (("w64", 2), ("w64c1", 1)) for bits in (8, 4)}
    out["paired_release_plus_w64_int8"] = state_bytes(2, 8, paired_with=2)
    out["paired_release_plus_w64c1_int8"] = state_bytes(1, 8, paired_with=2)
    ok = out["w64_int8"]["total"] == 4554 and out["w64_int4"]["total"] == 3018 and out["w64c1_int8"]["total"] == 3134 \
        and out["w64c1_int4"]["total"] == 2110
    return ok, out


def run(checks, out_name):
    res = {}
    for name, f in checks:
        try:
            ok, info = f()
        except Exception as error:   # a failed proof, not a crash
            ok, info = False, f"{type(error).__name__}: {error}"
        res[name] = {"ok": bool(ok), "info": info}
        print(f"[sm101-check] {name} {'OK' if ok else 'FAIL'}: {json.dumps(info)[:900]}")
    allok = all(v["ok"] for v in res.values())
    res["all_ok"] = allok
    (HERE / out_name).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("[sm101-check]", "ALL OK" if allok else "FAILED", f"-> {out_name}")
    return 0 if allok else 1


def main():
    if sys.argv[1:] == ["cooldown"]:
        return run((("C1_cooldown_from_e38", cooldown_checks),), "check-cpu-sm101-cooldown.json")
    return run((("K1_code", k1), ("K2_options", k2), ("K3_epochs", k3), ("K4_cut", k4), ("K5_cooldown_recipe", k5),
                ("K6_rule", k6), ("K7_breaks", k7), ("K8_bytes", k8)), "check-cpu-sm101.json")


if __name__ == "__main__":
    raise SystemExit(main())
