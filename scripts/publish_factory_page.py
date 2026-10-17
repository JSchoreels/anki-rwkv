#!/usr/bin/env python
"""Build pages/data/factory.json (read by pages/factory.html) from the hand-edited curated file and live state.

Inputs (all read-only):
  pages/data/factory-curated.json   headline cards, NOW notes, PLAN rows (optional `match`), BACKLOG, RESULTS, known labels
  scripts/r01_queue.py status       running / waiting tickets of owner small-model
  artifacts/coordination/queue/     ticket files (start time, expected minutes) and history.jsonl (finished tickets)
  <worktree>/artifacts/small-model/runbook-*.log   newest log: latest training progress line
  <worktree>/artifacts/small-model/heldout.jsonl   464-user scoring rows -> new "pending" result rows

Only the output file is written. It is rewritten only when something material changed (running / waiting / recently
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
HISTORY_DAYS = 3  # plan rows only look at tickets finished this recently
PROGRESS_FRESH_SECONDS = 20 * 60

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
            rows.append({"ticket": r["ticket"], "purpose": clean(r.get("purpose", "")), "finished": local(r["finished"]),
                         "started": local(r["started"]) if r.get("started") else None, "exit": r.get("exit_code"),
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
            "date": local(first).strftime("%m-%d"), "kind": "gpu-score", "name": html.escape(label[:-len("/w8+s8")]), "id": "",
            "label": label, "auto": True, "first_seen": first,
            "curve": round(c[0], 5), "instant": round(c[1], 5),
            "curve_pct": round((c[0] / f[0] - 1) * 100, 2), "instant_pct": round((c[1] / f[1] - 1) * 100, 2),
            "eval": f"{USERS} users, {reviews / 1e6:.2f}M reviews", "fork": f"fork {f[0]:.5f} / {f[1]:.5f}",
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
    """
    matches = row.get("match") or []
    if isinstance(matches, str):
        matches = [matches]
    if not matches:
        return {}
    cutoff = now() - timedelta(days=HISTORY_DAYS)
    done = []
    for m in matches:
        hit = [t for t in finished if m in t["purpose"] and t["finished"] >= cutoff]
        done.append(hit[-1] if hit else None)
    got = [d for d in done if d]
    first = min([(d["started"] or d["finished"]) for d in got], default=None)
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
    if got and len(got) == len(matches):
        last = max(got, key=lambda d: d["finished"])
        return {"state": "done", "ended": clock(last["finished"]), "exit": max((d["exit"] or 0) for d in got),
                "started": clock(first), "started_at": iso(first), "ended_at": iso(last["finished"])}
    if got:
        return {"state": "partial", "ended": clock(max(d["finished"] for d in got)),
                "ended_at": iso(max(d["finished"] for d in got))}
    return {"state": "planned"}


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
        row = {"what": note.get("what") or src.get("what") or html.escape(t["purpose"]), "id": ident,
               "kind": note.get("kind") or src.get("kind") or "gpu-train",
               "decides": note.get("decides") or src.get("decides") or "", "why": note.get("why") or src.get("why") or "",
               "job": html.escape(t["purpose"]), "linked": linked, "track": note.get("track") or src.get("track") or TRACK,
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

    recent = [{"ticket": t["ticket"], "id": plan_id(t["purpose"]), "what": html.escape(t["purpose"]),
               "ended": clock(t["finished"]), "date": t["finished"].strftime("%m-%d"), "exit": t["exit"],
               "ran": hms(t["ran"]) if t["ran"] else ""}
              for t in reversed(finished[-RECENT:])]
    waiting_rows = [{"ticket": t["ticket"], "id": plan_id(t["purpose"]), "what": html.escape(t["purpose"]),
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="print what would change, write nothing")
    parser.add_argument("--force", action="store_true", help="write even when only progress changed")
    args = parser.parse_args()
    previous = read_json(OUTPUT)
    data = build(previous)
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
