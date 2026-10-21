#!/usr/bin/env bash
# SM55b (2026-10-21 evening): relaunch of runbook-2026-10-21-sm55.sh after its decay ticket (000596) stopped after 2 h 14 min
#   with "Unlabelled batch user 5599, offset 32768": user 5599 (32,782 reviews) ends with 14 first reviews (no label), so
#   with 32,768-review updates its 2nd update had no label. The replay export is complete and correct (all 8,000 files
#   have rows == reviews and label counts == the export records; scripts/check_replay_label_windows.py). Fix: that last
#   unlabelled window is cut from the user's train_range ([0, 32768]); it carries no loss and feeds no later review
#   (a one-user GPU run of the cut row passed, ticket 000601). The cut is recorded in decay.json
#   ("trimmed_unlabelled_tails"; the original rows stay in protocol.untrimmed.json, which the SM36 equivalence check reads).
#   New step before the decay: check_replay_label_windows.py (every user's replay file complete, every update labelled).
#   Link/export/verify are done (status must still say complete); the failed attempt is in <decay>/failed-000596-*.
# Original header follows.
# SM55 (2026-10-21): SM36's recipe with ONE change, the 8k-user final tier instead of the 5k gate tier.
# SM36 = runbook-2026-10-19-sm42-sm36.sh ARM=sm36 (wt/sm42, read only): one-epoch final decay at 32,768 x 1, strided-lane
#   carried state, replay (full-history fork) KD targets, Muon "expertium-iter53" with a fresh optimizer state,
#   gpu_memory_fraction 0.8, from scale-w64kd-gate-sc-lanes-cont epoch 12 (wt/sm21v2, read only).
# SM55 = the same, trained on all 8,000 training users of the 8k split (accuracy-loop tier "final", 581,942,738 reviews,
#   1.575x SM36's), via `expertium_decay.py prepare --train-tier final`. Shuffle seed 1 as SM36 (the order is the shuffle
#   of the 8k list). The decay covers one epoch of the 8k users (LR 7e-4 -> 0 one_minus_sin over all of them).
#   Consequence of the one change: replay targets are needed for the 3,000 users outside the 5k export. They go to a NEW
#   folder kd-replay/fork-fp32-final8k = hard links to the 5,000 files of fork-fp32-gate5k (same bytes, that folder is
#   untouched) + the 3,000 new users exported with the same SM11 tool and settings (the manifest must match).
#   So the decay options differ from SM36's in distillation.replay (folder path) only.
# Steps (one detached chain): CPU checks -> link folder -> prepare + options check vs SM36's decay.json/protocol.json ->
#   GPU export tickets (resumable, 95 min, --time-limit 5100) -> status complete + `verify` on 8 new users (CPU) ->
#   GPU decay (smoke first, one ticket) -> GPU scoring of the 732 pool minus the 9 never-fit users (723; the 464 are a
#   subset) -> summary (paired vs SM36 w8+s8 and vs the fork, 464 and 732).
# Breaks: before every GPU ticket the chain computes from quiet.json AND the daily 21:00-01:00 rule (brief 2026-10-21)
#   whether the ticket's expected end is before the next break; if not, it sleeps to the break's end (01:00). The ticket
#   gets --max-wait-minutes so a queue wait cannot push it into a break (exit 75 -> sleep to the break end, retry).
#   CPU steps (verify, summary) also wait outside the breaks.
#
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/runbook-2026-10-21-sm55.sh
#   launch   nohup bash <this file> > C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/runbook-2026-10-21-sm55.log 2>&1 & disown
#   resume   rerun the launch command (exported users, the finished decay and scored users are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/sm55-STOP (checked before every ticket)
#   summary  SUMMARY_ONLY=1 bash <this file>
# Do not edit scripts/ expertium/ rwkv/ of the sm55 worktree while it runs (SourceWatch stops the epoch).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm55
SM=$WT/artifacts/small-model
V2W=C:/Users/jonat/workspace/wt/sm21v2                  # read only: source run (e12)
S42=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model   # read only: SM36 decay files and rows
SRC_WT=C:/Users/jonat/workspace/wt/small-model          # read only: fork rows (464 + 268 pool extra)
DEC_PY=$WT/scripts/expertium_decay.py
EXPORT=$WT/scripts/sm11_replay_export.py
SC=$WT/scripts/small_model_heldout.py
CKPT=expertium/E89-F65e-F77a/master.pth
SRC_EXP=$V2W/artifacts/accuracy/experiments/scale-w64kd-gate-sc-lanes-cont; FROM=12
SM36_DECAY=$S42/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon
SM36_464=$S42/heldout-sm36.jsonl; SM36_POOL=$S42/heldout-sm36-pools.jsonl; SM36_LBL=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon/w8+s8
FORK_464=$SRC_WT/artifacts/small-model/heldout.jsonl; FORK_POOL=$SRC_WT/artifacts/small-model/heldout-pools.jsonl
REPLAY5K=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k
REPLAY8K=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-final8k
RUN8K=$SM/sm55-final8k-run.json                         # export --run: train_users = the 8,000 final-tier users
USERS732=$SM/sm55-eval-users-732-fit.json               # 732 pool minus the 9 never-fit users (723)
D=$SM/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon-final8k
L=s-w64kd-final-sc-lanes-dec12-1ep32k-muon
OUT_POOL=$SM/heldout-sm55-pools.jsonl
SPEC='{"lanes": "strided"}'
DIST="{\"teacher\": \"C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY8K\"}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\"}"
MIN_EXPORT=95; LIMIT_EXPORT=5100; MAX_EXPORT_TICKETS=4   # ~220.6M replayed reviews; the 5k export ran 395M in 187 min
MIN_DECAY=420        # ticket 000596 ran 215.9M reviews in 8,012 s (26,946 reviews/s) -> 581.9M ~360 min + smoke/setup; +15% margin
MIN_SCORE1=45; MIN_SCORE2=20                            # SM36 732 pool: 18 min + 1.4 min resume
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION   # both default to a 0.8 cap
log() { echo "[sm55 $(date '+%m-%d %H:%M:%S')] $*"; }

window() {  # window MINUTES -> "GO <max wait minutes>" or "WAIT <seconds> <reason>"
  M="$1" Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os
from datetime import datetime, timedelta
now, need = datetime.now(), timedelta(minutes=float(os.environ["M"]))
import re
blocks = []
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"WAIT 600 quiet.json unreadable: {error}"); raise SystemExit
try:
    windows = [(w["start"], w["end"], w.get("scope", "all")) for w in json.loads(text)]
except (ValueError, KeyError, TypeError):
    # malformed JSON (e.g. unescaped quotes in a reason): take every start/end pair textually, scope assumed "all"
    windows = [(s, e, "all") for s, e in re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)]
for s, e, scope in windows:
    if scope in ("all", "gpu"):
        blocks.append((datetime.fromisoformat(s), datetime.fromisoformat(e), "quiet.json"))
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
for d in range(-1, 9):  # the brief's daily break 21:00-01:00
    start = day + timedelta(days=d, hours=21)
    blocks.append((start, start + timedelta(hours=4), "daily 21:00-01:00"))
blocks = sorted(b for b in blocks if b[1] > now)
merged = []
for s, e, why in blocks:
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e), merged[-1][2])
    else:
        merged.append((s, e, why))
for s, e, why in merged:
    if now >= s or now + need > s:
        print(f"WAIT {int((e - now).total_seconds()) + 60} break {s:%m-%d %H:%M}-{e:%m-%d %H:%M} ({why}); "
              f"a {need.total_seconds() / 60:.0f}-min ticket would end {now + need:%m-%d %H:%M}"); raise SystemExit
    print(f"GO {max(1, int((s - need - now).total_seconds() // 60))}"); raise SystemExit
print("GO 100000")
PYEOF
}

wait_window() {  # wait_window MINUTES WHAT: sleep until a MINUTES-long step ends before the next break; sets MAXWAIT
  while true; do
    [ -e "$SM/sm55-STOP" ] && { log "STOP file present; stop"; return 1; }
    local w; w=$(window "$1")
    case "$w" in
      GO*) MAXWAIT=${w#GO }; return 0;;
      WAIT*) set -- "$1" "$2" "$w"; local s; s=$(echo "$3" | cut -d' ' -f2)
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait: ${3#WAIT }"; MAXWAIT=1; return 0; fi
             log "$2 waits: ${3#WAIT * }"; sleep "$s";;
      *) log "window check failed: $w"; sleep 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...   (window-aware; retries after a break if the queue wait would run into it)
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_window "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min, max wait $MAXWAIT min)"; echo "    cmd: $*"; return 0; fi
    log "enqueue: $purpose (expected $minutes min, max wait $MAXWAIT min)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$MAXWAIT" --launched-by "detached runbook runbook-2026-10-21-sm55b.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_window "$1" "$2"; }   # CPU work also stays out of the breaks
need() { if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): $1"; return 0; fi; log "STOP: $1"; return 1; }
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }

preconditions() {
  [ -f "$SRC_EXP/run/epoch-$FROM/$CKPT" ] || { need "source epoch $FROM checkpoint missing" || return 2; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)" || return 2; fi
  grep -q "ALL OK" "$SM/check-sm11-replay-targets-sm55.txt" 2>/dev/null || { need "SM11 CPU check (check-sm11-replay-targets-sm55.txt) not ALL OK" || return 2; }
  for f in "$SM36_DECAY/decay.json" "$SM36_DECAY/protocol.json" "$SM36_464" "$SM36_POOL" "$FORK_464" "$FORK_POOL" "$RUN8K" "$USERS732" \
           "$REPLAY5K/manifest.json"; do
    [ -f "$f" ] || { need "missing $f" || return 2; }
  done
  # the 8k tier: disjoint from the 464 and 732 pools and the sealed users; overlap with the 2,225 set reported
  WTR="$WT" RUN8K="$RUN8K" "$PY" -X utf8 - <<'PYEOF' || return 2
import json, os, sys
sys.path.insert(0, os.environ["WTR"])
from scripts.eval_pools import pool_users, split_info
_, split, tiers = split_info()
final = set(tiers["tiers"]["final"]["train_users"])
run = json.load(open(os.environ["RUN8K"]))
bad = []
if set(run["train_users"]) != final or len(final) != 8000:
    bad.append("sm55-final8k-run.json is not the final tier")
for p in (464, 732):
    if final & set(pool_users(p)):
        bad.append(f"final tier overlaps the {p} pool")
if final & set(split["final_test_users"]):
    bad.append("final tier overlaps the sealed final-test users")
if bad:
    print("[sm55-tier] REFUSE: " + "; ".join(bad)); raise SystemExit(1)
print(f"[sm55-tier] OK: 8,000 final-tier users ({tiers['tiers']['final']['train_reviews']:,} reviews), disjoint from the 464 and 732 "
      f"pools and the 500 sealed users; {len(final & set(pool_users(2225)))} of the 2,225 set are training users here (2,225 invalid for SM55)")
PYEOF
  free_gb=$(df -BG --output=avail "$(dirname "$REPLAY5K")" 2>/dev/null | tail -1 | tr -dc 0-9)
  log "free disk: ${free_gb:-?} GB (need ~2.2 GB for 3,000 users)"
  [ "${free_gb:-0}" -ge 10 ] || { need "less than 10 GB free" || return 2; }
  return 0
}

link_replay() {  # NEW folder = hard links to the 5,000 gate files (+ the same manifest); idempotent
  A="$REPLAY5K" B="$REPLAY8K" "$PY" -X utf8 - <<'PYEOF'
import os, shutil
from pathlib import Path
a, b = Path(os.environ["A"]), Path(os.environ["B"])
(b / "users").mkdir(parents=True, exist_ok=True)
if (b / "manifest.json").exists():
    if (b / "manifest.json").read_bytes() != (a / "manifest.json").read_bytes():
        raise SystemExit(f"{b}/manifest.json differs from {a}'s")
else:
    shutil.copyfile(a / "manifest.json", b / "manifest.json")
linked = kept = 0
for f in sorted((a / "users").glob("*.npz")):
    g = b / "users" / f.name
    if not g.exists():
        os.link(f, g); linked += 1
    elif not os.path.samefile(f, g):
        raise SystemExit(f"{g} exists and is not a hard link of {f}")
    else:
        kept += 1
print(f"[sm55-link] {b}: {linked} new hard links, {kept} present, from {a} (manifest identical)")
PYEOF
}

prepare_decay() {
  if [ "$DRYRUN" = 1 ]; then DUSE=$SM/cont-check-sm55-decay; rm -rf "$DUSE" "$SM/cont-check-sm55-gate"; else DUSE=$D; fi
  if [ ! -d "$DUSE" ]; then
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$FROM" --out "$DUSE" --fraction 1.0 --shape one_minus_sin \
      --evaluate none --seed 1 --fresh-optimizer --train-tier final --extra-options "$EXTRA" || { log "prepare failed"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ]; then  # self-test of --train-tier: the gate tier must reproduce SM36's users and order exactly
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$FROM" --out "$SM/cont-check-sm55-gate" --fraction 1.0 \
      --shape one_minus_sin --evaluate none --seed 1 --fresh-optimizer --train-tier gate --extra-options "$EXTRA" >/dev/null || return 1
  fi
  A="$SM36_DECAY" B="$DUSE" G="$SM/cont-check-sm55-gate" SRC_EXP="$SRC_EXP" FROM=$FROM R8="$REPLAY8K" R5="$REPLAY5K" \
    WTR="$WT" "$PY" -X utf8 - <<'PYEOF' || return 1
import json, os, random, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
a_dir, b_dir, g_dir = Path(os.environ["A"]), Path(os.environ["B"]), Path(os.environ["G"])
a = json.loads((a_dir / "decay.json").read_text()); b = json.loads((b_dir / "decay.json").read_text())
pa = json.loads((a_dir / "protocol.json").read_text())
pb = json.loads((b_dir / ("protocol.untrimmed.json" if (b_dir / "protocol.untrimmed.json").exists() else "protocol.json")).read_text())
pt = json.loads((b_dir / "protocol.json").read_text())
trims = {t["user_id"]: t["to"] for t in b.get("trimmed_unlabelled_tails", [])}
if [dict(r, train_range=[0, trims.get(r["user_id"], r["train_range"][1])]) for r in pb["rows"]] != pt["rows"] or any(k != "rows" and pb[k] != pt[k] for k in pb):
    print("[sm55-check] REFUSE: protocol.json differs from protocol.untrimmed.json beyond the recorded tail trims"); raise SystemExit(1)
if trims:
    print(f"[sm55-check] protocol.json = untrimmed rows with tail trims {trims} (recorded in decay.json)")
tiers = json.loads((Path(os.environ["WTR"]) / "artifacts/accuracy/tiers.json").read_text())
final = tiers["tiers"]["final"]
fail = []
for k in ("fraction_of_epoch_reviews", "shuffle_seed", "source_epoch", "source_case", "source_update_offset_end", "model", "track",
          "dropped_width32_options", "evaluation_set", "evaluation_users"):
    if a[k] != b[k]:
        fail.append(f"{k}: {a[k]} vs {b[k]}")
if (b["training_users"], b["training_reviews"]) != (8000, final["train_reviews"]) or b.get("train_users_from") != "tier final (artifacts/accuracy/tiers.json)":
    fail.append(f"training users/reviews {b['training_users']}/{b['training_reviews']} ({b.get('train_users_from')})")
if not a["lr"].startswith("7e-4 -> 0, shape one_minus_sin, over all") or a["lr"].split(" over ")[0] != b["lr"].split(" over ")[0]:
    fail.append(f"lr {a['lr']} vs {b['lr']}")
order = final["train_users"][:]; random.Random(1).shuffle(order)
if [r["user_id"] for r in pb["rows"]] != order:
    fail.append("protocol rows are not the seed-1 shuffle of the 8,000 final-tier users")
if any(r["train_range"] != [0, r["available_reviews"]] for r in pb["rows"]):
    fail.append("some rows do not train on the whole history")
if g_dir.exists():  # DRYRUN self-test: --train-tier gate == SM36's own rows (users, order, ranges)
    pg = json.loads((g_dir / "protocol.json").read_text())
    if pg["rows"] != pa["rows"]:
        fail.append("--train-tier gate does not reproduce SM36's protocol rows")
    else:
        print("[sm55-check] --train-tier gate reproduces SM36's 5,000 rows (users, order, ranges) exactly")
for k in ("full_run", "training", "dataset_path", "dataset_revision", "execution_mode", "evaluation_rows"):
    if pa[k] != pb[k]:
        fail.append(f"protocol {k} differs: {pa[k]} vs {pb[k]}")
oa, ob = json.loads(json.dumps(a["options"])), json.loads(json.dumps(b["options"]))
ra, rb = oa["distillation"].pop("replay"), ob["distillation"].pop("replay")
if Path(ra) != Path(os.environ["R5"]) or Path(rb) != Path(os.environ["R8"]):
    fail.append(f"replay folders {ra} / {rb}")
if oa != ob:
    fail.append("options differ from SM36 beyond the replay folder: "
                f"{ {k: (oa.get(k), ob.get(k)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)} }")
if b["source_run"] != a["source_run"]:
    fail.append(f"source run {b['source_run']}")
if fail:
    print("[sm55-check] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[sm55-check] OK vs SM36's decay.json/protocol.json: identical except the users (8,000 final tier, {b['training_reviews']:,} reviews, "
      f"{b['training_reviews'] / a['training_reviews']:.3f}x SM36) and distillation.replay ({Path(rb).name}); same source e{b['source_epoch']}, "
      f"fresh Muon state, full_run/LR schedule/training/dataset identical, {b['lr']}")
PYEOF
  DEC="$DUSE" ROOT_WT="$WT" "$PY" -X utf8 - <<'PYEOF' || { log "allowlist/replay check failed"; return 1; }
import json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.environ["ROOT_WT"])
o = json.loads((Path(os.environ["DEC"]) / "decay.json").read_text())["options"]
text = (Path(os.environ["ROOT_WT"]) / "scripts" / "gpu_r01_prefetch.py").read_text(encoding="utf-8")
allow = set(re.findall(r'"([a-z_0-9]+)"', text[text.index("set(options) - {"):text.index("or bool(options.get(\"preprocessing_cache\"))")]))
extra = sorted(set(o) - allow)
if extra:
    print("options outside the prefetch allowlist:", extra); sys.exit(1)
from scripts.state_carry_train import validate_options
validate_options(o)
from scripts.expertium_options import ReplayTargets
rt = ReplayTargets(o["distillation"])
print(f"[check] state_carry options valid, replay spec valid ({rt.folder.name}, teacher sha256 {str(rt.teacher_sha256)[:12]}), "
      f"{len(o)} option keys allowlisted")
PYEOF
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DUSE" --out "$SM/check-muon-fresh-decay-sm55.json" \
    || { log "Muon fresh-state CPU check failed"; return 1; }
  if [ "$DRYRUN" = 1 ]; then rm -rf "$DUSE" "$SM/cont-check-sm55-gate"; fi
  return 0
}

export_replay() {
  local previous=-1 ST code done_users
  for ticket in $(seq 1 "$MAX_EXPORT_TICKETS"); do
    ST=$("$PY" -X utf8 "$EXPORT" status --out "$REPLAY8K" --run "$RUN8K"); code=$?
    if [ "$code" = 0 ]; then log "export complete: $ST"; return 0; fi
    done_users=$("$PY" -X utf8 -c "import json,sys; print(json.loads(sys.argv[1])['complete_users'])" "$ST" 2>/dev/null)
    log "export status: $ST"
    if [ "$DRYRUN" != 1 ] && [ "${done_users:-x}" = "$previous" ]; then log "STOP: no export progress in the last ticket (see $REPLAY8K/progress.jsonl)"; return 1; fi
    previous=${done_users:-x}
    q $MIN_EXPORT "SM55 replay targets, 3,000 final-tier users outside the 5k export (ticket $ticket, resumable)" -- \
      "$PY" -X utf8 "$EXPORT" export --out "$REPLAY8K" --run "$RUN8K" --device cuda --time-limit $LIMIT_EXPORT
    rc=$?
    if [ "$rc" != 0 ] && [ "$rc" != 3 ]; then log "STOP: export ticket exit $rc"; return 1; fi
    [ "$DRYRUN" = 1 ] && { log "DRYRUN: export arguments validated by status (exit $code)"; return 0; }
  done
  "$PY" -X utf8 "$EXPORT" status --out "$REPLAY8K" --run "$RUN8K" || { log "STOP: export incomplete after $MAX_EXPORT_TICKETS tickets (rerun resumes)"; return 1; }
}

verify_replay() {  # SM11 `verify` (CPU float32 recompute) on 8 of the NEW users (single pass, <= 20k reviews), seed 0
  local users
  users=$(RUN8K="$RUN8K" "$PY" -X utf8 -c "
import json, os, random
r = json.load(open(os.environ['RUN8K']))
c = r['review_counts']
pool = [u for u in r['extra_users_not_in_gate'] if c[str(u)] <= 20000]
print(','.join(map(str, sorted(random.Random(0).sample(pool, 8)))))")
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would verify users $users"; return 0; fi
  cpu_step 15 "verify" || return 1
  "$PY" -X utf8 "$EXPORT" verify --out "$REPLAY8K" --run "$RUN8K" --users "$users" || { log "STOP: verify failed (see $REPLAY8K/verify.json)"; return 1; }
  log "verify ok on new users $users"
}

label_windows() {  # every user's replay file complete, every update labelled; trims an unlabelled last window once
  local T="$D"
  if [ "$DRYRUN" = 1 ]; then T=$SM/cont-check-sm55b-labels; rm -rf "$T"; mkdir -p "$T"; cp "$D/decay.json" "$D/protocol.json" "$T/"
    [ -f "$D/protocol.untrimmed.json" ] && cp "$D/protocol.untrimmed.json" "$T/"; fi
  "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$T" --trim-unlabelled-tail || { log "STOP: replay label windows (trim pass) failed"; return 1; }
  "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$T" --out "$SM/check-replay-label-windows-sm55.json"     || { log "STOP: replay label windows check failed (see check-replay-label-windows-sm55.json)"; return 1; }
  log "label windows ok: every update of the 8,000 users has labels ($T)"
  [ "$DRYRUN" = 1 ] && rm -rf "$T"
  return 0
}

main_run() {
  "$PY" -X utf8 "$EXPORT" status --out "$REPLAY8K" --run "$RUN8K" >/dev/null || { log "STOP: replay targets incomplete (run runbook-2026-10-21-sm55.sh's export)"; return 1; }
  [ -d "$D" ] || { log "STOP: $D missing (prepared by runbook-2026-10-21-sm55.sh)"; return 1; }
  label_windows || return 1
  prepare_decay || return 1
  if [ "$DRYRUN" = 1 ] || [ ! -f "$D/$CKPT" ]; then
    q $MIN_DECAY "SM55 1-epoch decay at 32k on the 8k final tier (581.9M reviews), lanes carry, Muon fresh, from sc-lanes e12 (smoke first)" -- \
      "$BASH_EXE" -c "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" || { log "STOP: decay ticket failed"; return 1; }
  fi
  [ "$DRYRUN" = 1 ] || [ -f "$D/$CKPT" ] || { log "STOP: no decay checkpoint"; return 1; }
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUT_POOL" "$L/w8+s8")" -lt 723 ]; then
      q $((pass == 1 ? MIN_SCORE1 : MIN_SCORE2)) "SM55 scoring $L, 732 pool minus 9 never-fit (pass $pass)" -- "$PY" -X utf8 "$SC" --pool 732 \
        --users-file "$USERS732" --model "$L=expertium:w64:$D/$CKPT" --only "$L/w8+s8" --time-limit 4800 --out "$OUT_POOL" || { log "STOP: scoring ticket failed"; return 1; }
    fi
  done
  log "732-pool users scored: $(label_users "$OUT_POOL" "$L/w8+s8") of 723"
  return 0
}

summary() {
  WTR="$WT" SM="$SM" OUT_POOL="$OUT_POOL" L="$L/w8+s8" SM36_464="$SM36_464" SM36_POOL="$SM36_POOL" SM36_LBL="$SM36_LBL" \
    FORK_464="$FORK_464" FORK_POOL="$FORK_POOL" D="$D" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g
from scripts.eval_pools import pool_users

class Every:
    def __contains__(self, _):
        return True

def load(files, label):
    merged = {}
    for f in files:
        if Path(f).exists():
            merged.update(g.load(f, [label], Every())[label])
    return merged

def pair(name, pool, files_a, label_a, files_b, label_b, files_fork):
    """b - a, review-weighted, per head, pool users present in a, b and the fork; user-level bootstrap (2,000, seed 0)."""
    fork = "fork/w8+s8"
    per = {label_a: load(files_a, label_a), label_b: load(files_b, label_b), fork: load(files_fork, fork)}
    users = sorted(set(per[label_a]) & set(per[label_b]) & set(per[fork]) & set(pool_users(pool)))
    if not users:
        return name, None, f"{name}: no common users"
    res = {"users": len(users), "a": label_a, "b": label_b, "delta": "b - a"}
    for head, hname in g.HEADS.items():
        va, sizes = g.weighted(per, label_a, users, head)
        vb, _ = g.weighted(per, label_b, users, head)
        vf, _ = g.weighted(per, fork, users, head)
        ea, eb, ef = g.rw(va, sizes), g.rw(vb, sizes), g.rw(vf, sizes)
        idx = np.random.default_rng(0).integers(0, len(sizes), size=(2000, len(sizes)))
        w = sizes[idx]
        deltas = ((vb[idx] - va[idx]) * w).sum(1) / w.sum(1)
        res[hname] = {"a": ea, "b": eb, "fork": ef, "delta": eb - ea, "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                      "a_pct_vs_fork": (ea / ef - 1) * 100, "b_pct_vs_fork": (eb / ef - 1) * 100, "users_b_better": int((vb < va).sum())}
    c, i = res["curve"], res["instant"]
    line = (f"{name} ({len(users)} users): SM55 curve {c['b']:.5f} ({c['b_pct_vs_fork']:+.2f}% vs fork) instant {i['b']:.5f} ({i['b_pct_vs_fork']:+.2f}%); "
            f"SM36 curve {c['a']:.5f} ({c['a_pct_vs_fork']:+.2f}%) instant {i['a']:.5f} ({i['a_pct_vs_fork']:+.2f}%); "
            f"SM55 - SM36 curve {c['delta']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] instant {i['delta']:+.5f} [{i['ci95'][0]:+.5f}, {i['ci95'][1]:+.5f}]; "
            f"fork curve {c['fork']:.5f} instant {i['fork']:.5f}")
    return name, res, line

e = os.environ
pairs = [pair("464 vs SM36", 464, [e["SM36_464"]], e["SM36_LBL"], [e["OUT_POOL"]], e["L"], [e["FORK_464"]]),
         pair("732 vs SM36", 732, [e["SM36_POOL"]], e["SM36_LBL"], [e["OUT_POOL"]], e["L"], [e["FORK_464"], e["FORK_POOL"]])]
result = Path(e["D"]) / "expertium/E89-F65e-F77a/result.json"
train = {}
if result.exists():
    r = json.loads(result.read_text())
    train = {k: r.get(k) for k in ("training_users", "updates", "review_presentations", "training_seconds", "reviews_per_second")}
sm = Path(e["SM"])
(sm / "sm55-summary.json").write_text(json.dumps({"pairs": {n: r for n, r, _ in pairs}, "training": train}, indent=1) + "\n")
tl = (f"Training: {train.get('training_users')} users, {train.get('review_presentations') or 0:,} reviews, {train.get('updates')} updates, "
      f"{(train.get('training_seconds') or 0) / 60:.0f} min, {train.get('reviews_per_second') or 0:,.0f} reviews/s") if train else "Training: result.json missing"
(sm / "sm55-summary.md").write_text(f"# SM55: {e['L']} vs SM36 {e['SM36_LBL']} (w8+s8, review-weighted, user bootstrap 95% CI; delta = SM55 - SM36)\n\n"
                                    + "\n\n".join(l for _, _, l in pairs) + f"\n\n{tl}\n")
print("[sm55-summary]\n" + "\n".join(l for _, _, l in pairs) + "\n" + tl)
PYEOF
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
if [ "${SUMMARY_ONLY:-0}" = 1 ]; then summary; exit 0; fi
rc=0
preconditions || rc=$?
[ $rc -eq 0 ] && { main_run || rc=$?; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then echo "DRYRUN OK"; else cpu_step 5 "summary" && summary; log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
