#!/usr/bin/env python
"""Build pages/data/factory.json (read by pages/factory.html) from the hand-edited curated file and live state.

Inputs (all read-only):
  pages/data/factory-curated.json   headline cards, NOW notes, PLAN rows (optional `match`), BACKLOG, RESULTS, known labels
  scripts/r01_queue.py status       running / waiting tickets of owner small-model
  artifacts/coordination/queue/     ticket files (start time, expected minutes) and history.jsonl (finished tickets)
  <worktree>/artifacts/small-model/runbook-*.log   newest log: latest training progress line
  <worktree>/artifacts/small-model/heldout.jsonl   464-user scoring rows -> new "pending" result rows

Sync audit: every run also lists where the hand-edited parts lag the queue and the results (sync_issues), keeps them in
artifacts/factory-page/sync-issues.json and appends new ones to the loop log as a "[sync] FAIL" line (not with --check).

The page file is rewritten only when something material changed (running / waiting / recently
finished tickets and their states, plan states, result labels, the curated file) or with --force, so a run that only
moves the progress percentage leaves the file (and git) alone. Exit code: 0 = file written (or would be, with
--check), 3 = nothing material changed.

usage: publish_factory_page.py [--check] [--force]
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys

EPOCH_TICKET = re.compile(r"epoch \d+/\d+")  # training tickets that run one epoch of several

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
CURATED = ROOT / "pages" / "data" / "factory-curated.json"
OUTPUT = ROOT / "pages" / "data" / "factory.json"
QUEUE_DIR = ROOT / "artifacts" / "coordination" / "queue"
WORKTREE = Path("C:/Users/jonat/workspace/wt/small-model")
SM = WORKTREE / "artifacts" / "small-model"
HELDOUT = SM / "heldout.jsonl"
OWNER = "small-model"
FORK = "fork/w8+s8"
USERS = 464
TRACK = "small model"
RECENT = 6  # finished tickets shown / hashed
PROGRESS_FRESH_SECONDS = 20 * 60
LOOP_LOG = ROOT / "artifacts" / "factory-page" / "loop.log"
SYNC_FILE = ROOT / "artifacts" / "factory-page" / "sync-issues.json"
SYNC_GRACE_HOURS = 2  # a finished experiment gets this long to receive its results row before it is reported
CLOSED = ("done", "hold", "dropped")  # curated plan states the queue does not override (running / waiting still do)

# Gate math of wt/small-model scripts/small_model_gate.py (copied): review-weighted benchmark LogLoss per head.
HEADS = {"ahead": "curve", "instant": "instant"}

PROGRESS = re.compile(
    r"\[[#.]*\]\s+(?P<pct>[\d.]+)%\s+(?P<done>\d+)/(?P<total>\d+) users\s+[\d.]+M \|\s+(?P<avg>[\d,]+) rev/s avg\s+"
    r"(?P<now>[\d,]+) now \|\s+(?P<elapsed>\d+)s ETA (?P<eta>\d+)s")
RUNNING = re.compile(r"^\s+(?P<ticket>\d{6}-\S+) running (?P<age>\d+:\d\d:\d\d) \((?P<owner>[^:]+): (?P<purpose>.*)\); holds")
WAITING = re.compile(r"^\s+\d+\.\s+(?P<ticket>\d{6}-\S+) waiting(?P<flags>.*?) \((?P<owner>[^:]+): (?P<purpose>.*)\); needs")


def now():
    return datetime.now(timezone.utc).astimezone()


def local(iso):
    return datetime.fromisoformat(iso).astimezone()


def clock(moment):
    return moment.strftime("%H:%M")


def public(text):
    """Queue text as published: HTML-escaped, and the outside researcher's name (in config names such as
    expertium-iter53) replaced, since published pages never name them."""
    return html.escape(re.sub(r"(?i)expertium", "outside-research", text))


def clean(purpose):
    purpose = purpose.strip()
    return purpose[len(OWNER) + 2:] if purpose.startswith(OWNER + ": ") else purpose


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}"


# ----------------------------------------------------------------------------------------------- queue state

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def queue_state():
    """Running and waiting tickets of OWNER, from `r01_queue.py status` plus their ticket files."""
    try:
        text = subprocess.run([str(PY), "scripts/r01_queue.py", "status"], cwd=ROOT, capture_output=True, text=True,
                              timeout=60, encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return None
    running, waiting = [], []
    for line in text.splitlines():
        m = RUNNING.match(line)
        if m and m["owner"] == OWNER:
            h, mi, s = (int(x) for x in m["age"].split(":"))
            record = read_json(QUEUE_DIR / "tickets" / f"{m['ticket']}.json") or {}
            started = local(record["started"]) if record.get("started") else now() - timedelta(hours=h, minutes=mi, seconds=s)
            running.append({"ticket": m["ticket"], "purpose": clean(m["purpose"]), "started": started,
                            "elapsed": h * 3600 + mi * 60 + s, "expected_minutes": record.get("expected_minutes")})
            continue
        m = WAITING.match(line)
        if m and m["owner"] == OWNER:
            waiting.append({"ticket": m["ticket"], "purpose": clean(m["purpose"]), "flags": m["flags"].strip(", ")})
    return running, waiting


def finished_tickets():
    """Finished tickets of OWNER from the queue ledger (history.jsonl), oldest first."""
    rows = []
    path = QUEUE_DIR / "history.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("owner") == OWNER and r.get("state") == "finished" and r.get("finished"):
            code = r.get("exit_code")
            if code == 3 and EPOCH_TICKET.search(r.get("purpose", "")):
                code = 0  # accuracy_loop.py train: exit 3 = epoch done, more epochs to do (not a failure)
            rows.append({"ticket": r["ticket"], "purpose": clean(r.get("purpose", "")), "finished": local(r["finished"]),
                         "started": local(r["started"]) if r.get("started") else None, "exit": code,
                         "ran": r.get("ran_seconds")})
    rows.sort(key=lambda r: r["finished"])
    return rows


def progress():
    """Latest progress line of the newest runbook log, when that log was written in the last minutes."""
    logs = sorted(SM.glob("runbook-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None
    log = logs[0]
    age = now().timestamp() - log.stat().st_mtime
    if age > PROGRESS_FRESH_SECONDS:
        return None
    with log.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        tail = handle.read().decode("utf-8", errors="replace")
    found = None
    for part in re.split(r"[\r\n]+", tail):
        for m in PROGRESS.finditer(part):
            found = m
    if not found:
        return None
    return {"fraction": float(found["pct"]) / 100, "users_done": int(found["done"]), "users_total": int(found["total"]),
            "rev_s_avg": int(found["avg"].replace(",", "")), "rev_s_now": int(found["now"].replace(",", "")),
            "eta_seconds": int(found["eta"]), "log": log.name}


# ----------------------------------------------------------------------------------------------- results

def heldout_results():
    """{label: {user_id: {head: (logloss, size)}}} for w8+s8 labels, last row per user wins."""
    per = {}
    try:
        lines = HELDOUT.read_text(encoding="utf-8").splitlines()
    except OSError:
        return per
    for line in lines:
        if "/w8+s8" not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:  # a line still being written
            continue
        label, bench = row.get("label", ""), row.get("benchmark") or {}
        if label.endswith("/w8+s8") and "ahead" in bench and "instant" in bench:
            per.setdefault(label, {})[row["user_id"]] = {h: (bench[h]["LogLoss"], bench[h]["size"]) for h in HEADS}
    return per


def weighted(per, label, users, head):
    values = [per[label][u][head][0] for u in users]
    sizes = [per[label][u][head][1] for u in users]
    return sum(v * s for v, s in zip(values, sizes)) / sum(sizes)


def new_result_rows(curated, previous):
    per = heldout_results()
    fork = per.get(FORK, {})
    if len(fork) < USERS:
        return []
    handled = {r["label"] for r in curated["results"] if r.get("label")} | set(curated.get("known_labels", []))
    seen = {r["label"]: r.get("first_seen") for r in (previous or {}).get("results", []) if r.get("auto")}
    rows = []
    for label in sorted(per):
        users = sorted(set(per[label]) & set(fork))
        if label == FORK or label in handled or len(per[label]) != USERS or len(users) != USERS:
            continue
        c = [weighted(per, label, users, h) for h in HEADS]
        f = [weighted(per, FORK, users, h) for h in HEADS]
        reviews = sum(per[label][u]["ahead"][1] for u in users)
        first = seen.get(label) or now().isoformat(timespec="seconds")
        rows.append({
            "date": local(first).strftime("%m-%d"), "kind": "gpu-score", "name": public(label[:-len("/w8+s8")]), "id": "",
            "label": label, "auto": True, "first_seen": first,
            "curve": round(c[0], 5), "instant": round(c[1], 5),
            "curve_pct": round((c[0] / f[0] - 1) * 100, 2), "instant_pct": round((c[1] / f[1] - 1) * 100, 2),
            "eval": f"{USERS} users, {reviews / 1e6:.2f}M reviews", "fork": f"big model {f[0]:.5f} / {f[1]:.5f}",
            "takeaway": "pending", "track": TRACK})
    rows.sort(key=lambda r: r["first_seen"], reverse=True)
    return rows


# ----------------------------------------------------------------------------------------------- assembly

def iso(moment):
    return moment.isoformat(timespec="seconds")


def plan_state(row, running, waiting, finished):
    """State of a plan row from the queue, with actual times (ISO) for the browser to lay out the timetable.

    done: started_at / ended_at (first start, last end). running: started_at and ends_at (row start + the row's
    `minutes`, else the ticket's expected minutes). Nothing here changes every cycle except through a state change.
    All ticket history counts (a done row stays done), except tickets finished before the row's `since` date, for a
    new row whose match text an older ticket also contains. A curated `state` in CLOSED (with its own started_at /
    ended_at) wins over the history; only a running or waiting ticket overrides it.
    """
    matches = row.get("match") or []
    if isinstance(matches, str):
        matches = [matches]
    if not matches:
        return {}
    cutoff = since_of(row)
    done, starts = [], []
    for m in matches:
        hit = [t for t in finished if m in t["purpose"] and (cutoff is None or t["finished"] >= cutoff)]
        done.append(hit[-1] if hit else None)
        starts += [t["started"] or t["finished"] for t in hit]
    got = [d for d in done if d]
    first = min(starts, default=None)  # the row's first ticket (a row of several tickets, e.g. one per epoch)
    for m in matches:
        r = next((t for t in running if m in t["purpose"]), None)
        if r:
            start = min(first, r["started"]) if first else r["started"]
            minutes = row.get("minutes") or r.get("expected_minutes")
            out = {"state": "running", "started": clock(r["started"]), "ticket": r["ticket"], "started_at": iso(start)}
            if minutes:
                out["ends_at"] = iso(start + timedelta(minutes=minutes))
            return out
    w = next((t for m in matches for t in waiting if m in t["purpose"]), None)
    if w:
        return {"state": "waiting", "ticket": w["ticket"]}
    if row.get("state") in CLOSED and not (row["state"] == "done" and not row.get("ended_at") and got):
        return {}  # a curated done row without its own times takes them from the tickets below
    if row.get("state") == "done" or got and len(got) == len(matches):
        last = max(got, key=lambda d: d["finished"])
        return {"state": "done", "ended": clock(last["finished"]), "exit": max((d["exit"] or 0) for d in got),
                "started": clock(first), "started_at": iso(first), "ended_at": iso(last["finished"])}
    if got:
        return {"state": "partial", "ended": clock(max(d["finished"] for d in got)),
                "ended_at": iso(max(d["finished"] for d in got))}
    return {"state": "planned"}


def since_of(row):
    """A row's `since` ("10-26" = this year, or "2026-10-26") as local midnight, else None."""
    text = row.get("since")
    if not text:
        return None
    if len(text) == 5:
        text = f"{now().year}-{text}"
    return datetime.fromisoformat(text).replace(tzinfo=now().tzinfo)


def plan_for(purpose, plan_rows):
    """The curated plan row whose `match` substring occurs in a queue purpose, else None."""
    for row in plan_rows:
        matches = row.get("match") or []
        if isinstance(matches, str):
            matches = [matches]
        if any(m and m in purpose for m in matches):
            return row
    return None


def build(previous):
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    queue = queue_state()
    if queue is None:
        raise SystemExit("r01_queue.py status failed")
    running, waiting = queue
    finished = finished_tickets()
    prog = progress() if running else None

    notes = curated.get("now_notes", [])
    now_rows = []
    for t in running:
        note = next((n for n in notes if n.get("match") and n["match"] in t["purpose"]), {})
        # A running ticket without its own now_notes entry takes What / Decides from the plan row it came from.
        src = plan_for(t["purpose"], curated["plan"]) or {}
        linked = bool(note or src)
        ident = note.get("id") or (f"{src['id']}, ticket {t['ticket'][:6]}" if src.get("id") else t["ticket"])
        row = {"what": note.get("what") or src.get("what") or public(t["purpose"]), "id": ident,
               "kind": note.get("kind") or src.get("kind") or "gpu-train",
               "decides": note.get("decides") or src.get("decides") or "", "why": note.get("why") or src.get("why") or "",
               "job": public(t["purpose"]), "linked": linked, "track": note.get("track") or src.get("track") or TRACK,
               "ticket": t["ticket"], "started": clock(t["started"]), "elapsed": hms(t["elapsed"]), "progress": None}
        end = None
        if len(running) == 1 and prog:
            row["progress"] = round(prog["fraction"], 4)
            row["rev_s"] = prog["rev_s_avg"]
            row["rev_s_avg"] = prog["rev_s_avg"]
            end = now() + timedelta(seconds=prog["eta_seconds"])
        elif t.get("expected_minutes"):
            end = t["started"] + timedelta(minutes=t["expected_minutes"])
        row["eta"] = "~" + clock(end) if end else ""
        now_rows.append(row)
    for n in notes:
        if n.get("manual"):
            now_rows.append({k: n.get(k, "") for k in ("what", "id", "kind", "why", "track", "started")}
                             | {"eta": n.get("eta", ""), "progress": None, "manual": True})

    plan = []
    for row in curated["plan"]:
        item = {k: v for k, v in row.items() if k != "match"} | plan_state(row, running, waiting, finished)
        plan.append(item)

    def plan_id(purpose):
        return (plan_for(purpose, curated["plan"]) or {}).get("id", "")

    recent = [{"ticket": t["ticket"], "id": plan_id(t["purpose"]), "what": public(t["purpose"]),
               "ended": clock(t["finished"]), "date": t["finished"].strftime("%m-%d"), "exit": t["exit"],
               "ran": hms(t["ran"]) if t["ran"] else ""}
              for t in reversed(finished[-RECENT:])]
    waiting_rows = [{"ticket": t["ticket"], "id": plan_id(t["purpose"]), "what": public(t["purpose"]),
                     "flags": t["flags"]} for t in waiting]

    results = new_result_rows(curated, previous) + curated["results"]
    material_src = {
        "running": [(t["ticket"], "running") for t in running], "waiting": [(t["ticket"], "waiting") for t in waiting],
        "finished": [(t["ticket"], t["exit"]) for t in finished[-RECENT:]],
        "plan": [(p.get("what"), p.get("state"), p.get("exit"), p.get("started_at"), p.get("ended_at")) for p in plan],
        "labels": [r["label"] for r in results if r.get("label")],
        "curated": hashlib.sha256(CURATED.read_bytes()).hexdigest(),
    }
    material = hashlib.sha256(json.dumps(material_src, sort_keys=True).encode()).hexdigest()[:16]

    sync = sync_issues(curated, {"plan": plan, "results": results, "backlog": curated["backlog"], "now": now_rows,
                                 "waiting": waiting_rows, "headline": curated.get("headline", [])},
                       running, waiting, finished)

    parts = [f"running {t['ticket'][:6]}" for t in running] or ["idle"]
    auto = [r for r in results if r.get("auto")]
    if auto:
        parts.append(f"{len(auto)} pending result{'s' if len(auto) > 1 else ''}")
    if finished and (previous or {}).get("recent") and finished[-1]["ticket"] != previous["recent"][0]["ticket"]:
        parts.append(f"finished {finished[-1]['ticket'][:6]}")
    stamp = now()
    return {
        "generated_at": stamp.isoformat(timespec="seconds"), "updated": stamp.strftime("%Y-%m-%d %H:%M"),
        "material": material, "summary": ", ".join(parts),
        "headline": curated.get("headline", []), "now": now_rows, "waiting": waiting_rows, "recent": recent,
        "plan": plan, "gpu_break": curated.get("gpu_break"), "backlog": curated["backlog"], "results": results,
        "_sync": sync,
    }


# ----------------------------------------------------------------------------------------------- sync audit

def ids_of(text):
    return [s.strip() for s in (text or "").split(",") if s.strip()]


def plain(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text or ""))


def mentions(ident, text):
    return re.search(rf"(?<![\w-]){re.escape(ident)}(?![\w-])", text) is not None


def eval_key(text):
    """(users, M reviews) of an eval string such as "723 users (732 set ...), 30.48M reviews"."""
    u = re.search(r"([\d,]+) users", text or "")
    m = re.search(r"([\d.]+)M ", (text or "") + " ")
    return (u[1].replace(",", ""), m[1]) if u and m else None


def sync_issues(curated, data, running, waiting, finished):
    """Where the hand-edited parts of the page lag the queue and the results. One line per issue.

    - a plan row still planned / partly done while its ID has a results row (dated on or after the row's `since`)
    - a plan row marked done without ended_at (the timetable would lay it out as upcoming)
    - an experiment whose plan rows are all closed for SYNC_GRACE_HOURS (within the last week) with no results row
      dated on or after its end, while no ticket of it runs or waits
    - a backlog item still idea / next / parked while its ID has a results row (on or after the item's `since`)
    - a now note that matches no running, waiting or recently finished ticket
    - a running or waiting ticket that matches no plan row or now note
    - a headline card whose value is not its "Best" row, or with a result row that beats "Best" on the same users
    - a pending (auto) result row older than a day
    """
    stamp = now()
    dated = {}
    for r in data["results"]:
        for i in ids_of(r.get("id")):
            dated.setdefault(i, []).append(r.get("date", ""))

    def latest(ident, since):
        return max((d for d in dated.get(ident, []) if not since or d >= since), default=None)

    def mmdd(text):
        return text[-5:] if text else ""

    issues = []
    live = [t["purpose"] for t in running + waiting]
    still_open = {i for p in data["plan"] if p.get("state") not in CLOSED for i in ids_of(p.get("id"))}
    ends = {}
    for p in data["plan"]:
        state = p.get("state") or "planned"
        name = f'plan row "{plain(p["what"])[:60]}" ({p.get("id") or "no ID"})'
        if state in ("planned", "partial"):
            for i in ids_of(p.get("id")):
                d = latest(i, mmdd(p.get("since")))
                if d:
                    issues.append(f"{name} is {state}, but {i} has a results row dated {d}: close it "
                                  f"(state done / hold / dropped) or give it a since date")
        if state == "done" and not p.get("ended_at"):
            issues.append(f"{name} is done without ended_at")
        if state == "done" and p.get("ended_at"):
            for i in ids_of(p.get("id")):
                ends[i] = max(ends.get(i, ""), p["ended_at"])
    for i, end in sorted(ends.items()):
        age = stamp - local(end)
        if i in still_open or any(mentions(i, s) for s in live) or not timedelta(hours=SYNC_GRACE_HOURS) < age < timedelta(days=7):
            continue
        if not latest(i, end[5:10]):
            issues.append(f"{i} finished {end[5:16].replace('T', ' ')} and has no Factory results row dated on or after it")
    for b in data["backlog"]:
        if b.get("status") in ("idea", "next", "parked"):
            for i in ids_of(b.get("id")):
                d = latest(i, mmdd(b.get("since")))
                if d:
                    issues.append(f'backlog {i} "{plain(b["item"])[:60]}" is still {b["status"]}, but has a results row '
                                  f"dated {d}: mark it done or give it a since date")
    recent = [t["purpose"] for t in finished if stamp - t["finished"] < timedelta(days=2)]
    for n in curated.get("now_notes", []):
        if n.get("match") and not n.get("manual") and not any(n["match"] in s for s in live + recent):
            issues.append(f'now note "{n["match"][:60]}" matches no running, waiting or recent ticket: remove it')
    for r in data["now"]:
        if not r.get("manual") and not r.get("linked"):
            issues.append(f"running ticket {r['ticket'][:6]} has no plan row or now note: {plain(r['job'])[:90]}")
    for w in data["waiting"]:
        if not w["id"]:
            issues.append(f"queued ticket {w['ticket'][:6]} has no plan row: {plain(w['what'])[:90]}")
    for h in data["headline"]:
        best = next((c for c in h.get("compare") or [] if c.get("variant", "").startswith("Best")), None)
        if not best or best.get("curve") is None:
            continue
        shown = [float(x) for x in re.findall(r"\d+\.\d+", h.get("v", ""))[:2]]
        if len(shown) < 2 or abs(shown[0] - best["curve"]) > 5e-6 or abs(shown[1] - best["instant"]) > 5e-6:
            issues.append(f'headline "{plain(h["k"])[:50]}" shows {h.get("v")}, its Best row is {best["curve"]} / {best["instant"]}')
        key = eval_key(best.get("eval"))
        for r in data["results"]:
            if (key and r.get("curve") is not None and r.get("instant") is not None and eval_key(r.get("eval")) == key
                    and r["curve"] < best["curve"] - 1e-5 and r["instant"] < best["instant"] - 1e-5):
                issues.append(f'headline "{plain(h["k"])[:50]}": results row {r.get("id") or r.get("label")} ({r.get("date")}) '
                              f"beats its Best on both heads ({r['curve']} / {r['instant']})")
    for r in data["results"]:
        if r.get("auto") and stamp - local(r["first_seen"]) > timedelta(days=1):
            issues.append(f"pending result {r['label']} since {r['first_seen'][:16]} has no curated row")
    return issues


def report_sync(issues):
    """Keep the open issues (with first-seen times) in SYNC_FILE; append the new ones to the loop log as one
    "[sync] FAIL" line, which wakes the watching session's event waiter."""
    old = read_json(SYNC_FILE) or {}
    stamp = now().isoformat(timespec="seconds")
    SYNC_FILE.write_text(json.dumps({i: old.get(i, stamp) for i in issues}, indent=1, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    new = [i for i in issues if i not in old]
    if new:
        with LOOP_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now():%Y-%m-%d %H:%M:%S}] [sync] FAIL: {len(new)} new page sync issue(s), {len(issues)} open "
                         f"(artifacts/factory-page/sync-issues.json): {' | '.join(new)[:400]}\n")
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="print what would change, write nothing")
    parser.add_argument("--force", action="store_true", help="write even when only progress changed")
    args = parser.parse_args()
    previous = read_json(OUTPUT)
    data = build(previous)
    sync = data.pop("_sync")
    new = [] if args.check else report_sync(sync)
    print(f"SYNC {len(sync)} open page sync issue(s), {len(new)} new")
    for issue in sync:
        print(f"  sync: {issue}")
    changed = args.force or previous is None or previous.get("material") != data["material"]
    states = [f"{p['what'][:40]}: {p['state']}" for p in data["plan"] if p.get("state") in ("running", "waiting", "done")]
    print(f"material {data['material']} (was {previous.get('material') if previous else None}); {data['summary']}")
    print(f"running {len(data['now'])}, waiting {len(data['waiting'])}, recent {len(data['recent'])}, plan rows with state "
          f"{len(states)}, results {len(data['results'])} ({sum(bool(r.get('auto')) for r in data['results'])} pending)")
    for r in data["now"]:
        print(f"  now: {r['id']} started {r['started']} progress {r['progress']} eta {r['eta']}")
    for s in states:
        print(f"  plan {s}")
    if not changed:
        print("UNCHANGED (progress only)")
        return 3
    if args.check:
        print("WOULD WRITE " + str(OUTPUT))
        return 0
    OUTPUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE " + str(OUTPUT))
    print("SUMMARY " + data["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
