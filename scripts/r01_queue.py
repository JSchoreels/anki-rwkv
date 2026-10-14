"""FIFO run queue for sessions sharing this checkout: register a run, wait for your turn without polling, run, release.

Tickets are JSON files under artifacts/coordination/queue/tickets/. A ticket may start once every live ticket with a
lower sequence number that shares one of its claims has finished and the claim directories of scripts/r01_claim.py
are free; it then holds those claims until its command exits. Waiting blocks on a Windows directory-change
notification instead of polling, prints one progress line per --heartbeat-seconds, and removes tickets whose owning
process has died (a child command is bound to the wrapper's lifetime so a killed wrapper cannot leave a GPU job
behind). See docs/R01_PARALLEL_SESSIONS.md for the claim names and the session rules.

Typical use, as one background job:
    .venv/Scripts/python.exe scripts/r01_queue.py run --owner <session> --purpose "<what>" --claim gpu --claim training-source -- <command>
"""
from argparse import ArgumentParser, REMAINDER
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_claim import CLAIMS, COORDINATION, NAMES, holder, release as release_claim  # noqa: E402

QUEUE = COORDINATION / "queue"
TICKETS = QUEUE / "tickets"
STALE = QUEUE / "stale"
HISTORY = QUEUE / "history.jsonl"
COUNTER = QUEUE / "next-seq"
EXIT_WAIT_EXPIRED = 75  # EX_TEMPFAIL: the ticket gave up waiting; nothing ran
EXIT_CANCELLED = 76  # the ticket was cancelled (or otherwise removed) while waiting; nothing ran
CANCEL_GRACE_SECONDS = 5.0  # how long `cancel` lets a waiting wrapper notice before terminating it
STILL_ACTIVE = 259


def _utc():
    return datetime.now(timezone.utc)


def _iso(moment=None):
    return (moment or _utc()).isoformat(timespec="seconds")


def _ensure_dirs():
    for directory in (CLAIMS, TICKETS, STALE):
        directory.mkdir(parents=True, exist_ok=True)


def _retry(action, attempts=8, pause=0.05):
    """Windows readers briefly hold files open; retry the sharing violations instead of failing."""
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(pause)


def _read(path):
    for _ in range(8):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, ValueError):
            time.sleep(0.05)
    return None


def _public(record):
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _write_ticket(record):
    path = record["_path"]
    temporary = path.with_name(path.stem + ".tmp")
    temporary.write_text(json.dumps(_public(record)), encoding="utf-8")
    _retry(lambda: os.replace(temporary, path))


def _append_history(record):
    line = json.dumps(_public(record)) + "\n"

    def append():
        with HISTORY.open("a", encoding="utf-8") as handle:
            handle.write(line)

    try:
        _retry(append)
    except OSError:
        pass


# ----------------------------------------------------------------------------- process liveness

def _process_started(pid):
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _alive(pid, started=None):
    if not pid:
        return False
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            if started is not None and abs(process.create_time() - started) > 2.0:
                return False  # the PID was reused by another process
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except psutil.Error:
            pass
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


class _Watcher:
    """Block until something under the coordination directory changes (Windows); elsewhere sleep briefly."""

    def __init__(self, directory):
        self.directory = directory
        self.handle = None

    def __enter__(self):
        if os.name == "nt":
            kernel32 = ctypes.windll.kernel32
            kernel32.FindFirstChangeNotificationW.restype = ctypes.c_void_p
            kernel32.FindFirstChangeNotificationW.argtypes = [ctypes.c_wchar_p, ctypes.c_int, ctypes.c_uint32]
            kernel32.FindNextChangeNotification.argtypes = [ctypes.c_void_p]
            kernel32.FindCloseChangeNotification.argtypes = [ctypes.c_void_p]
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint32
            # FILE_NOTIFY_CHANGE_FILE_NAME | DIR_NAME | LAST_WRITE, whole subtree: tickets and claim directories.
            handle = kernel32.FindFirstChangeNotificationW(str(self.directory), True, 0x1 | 0x2 | 0x10)
            if handle not in (None, ctypes.c_void_p(-1).value):
                self.handle, self.kernel32 = handle, kernel32
        return self

    def wait(self, seconds):
        if self.handle is None:
            time.sleep(max(0.05, min(seconds, 15.0)))
            return
        result = self.kernel32.WaitForSingleObject(self.handle, int(max(0.05, seconds) * 1000))
        if result == 0:  # signalled: re-arm before the caller re-reads the queue so no change is missed
            self.kernel32.FindNextChangeNotification(self.handle)

    def __exit__(self, *exc):
        if self.handle is not None:
            self.kernel32.FindCloseChangeNotification(self.handle)
            self.handle = None


def _bind_child_lifetime(process):
    """Windows job object with kill-on-close: if this wrapper dies, its command dies with it (best effort)."""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.windll.kernel32

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BasicLimit(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32), ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        if not kernel32.AssignProcessToJobObject(job, int(process._handle)):
            return None
        return job  # keep the handle alive for the child's lifetime
    except Exception:
        return None


# ----------------------------------------------------------------------------- tickets

def _safe_name(owner):
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in owner) or "anon"


def _load():
    records = []
    for path in TICKETS.glob("*.json"):
        record = _read(path)
        if record and "seq" in record:
            record["_path"] = path
            records.append(record)
    return sorted(records, key=lambda record: record["seq"])


def _next_seq():
    try:
        return int(COUNTER.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return max([record["seq"] for record in _load()] + [0]) + 1


def _new_ticket(owner, purpose, claims, hold_minutes, command=None):
    _ensure_dirs()
    seq = _next_seq()
    while True:
        ticket = f"{seq:06d}-{_safe_name(owner)}"
        path = TICKETS / f"{ticket}.json"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            seq += 1
            continue
        record = {
            "ticket": ticket, "seq": seq, "owner": owner, "purpose": purpose, "claims": claims, "state": "queued",
            "created": _iso(), "expires": _iso(_utc() + timedelta(minutes=hold_minutes)), "pid": None,
            "pid_started": None, "heartbeat": _iso(), "started": None, "held": [], "command": command,
        }
        os.write(descriptor, json.dumps(record).encode("utf-8"))
        os.close(descriptor)
        try:
            COUNTER.write_text(str(seq + 1), encoding="utf-8")
        except OSError:
            pass
        record["_path"] = path
        return record


def _attach(ticket, owner):
    path = TICKETS / f"{ticket}.json"
    record = _read(path)
    if record is None:
        raise SystemExit(f"[queue] ticket {ticket} does not exist (expired, cancelled or finished)")
    if record.get("owner") != owner:
        raise SystemExit(f"[queue] ticket {ticket} belongs to {record.get('owner')}, not {owner}")
    if record.get("state") != "queued":
        raise SystemExit(f"[queue] ticket {ticket} is already {record.get('state')} (pid {record.get('pid')})")
    record["_path"] = path
    return record


def _retire(record, reason):
    """Remove a ticket, releasing any claims it holds, and keep its record under stale/ (or history for finished runs)."""
    for name in record.get("held") or []:
        current = holder(name)
        if current and current.get("ticket") == record["ticket"]:
            release_claim(name, current.get("owner"), force=True)
    record["retired"], record["reason"] = _iso(), reason
    path = record.get("_path")
    if path is not None:
        try:
            stale = STALE / path.name
            stale.write_text(json.dumps(_public(record)), encoding="utf-8")
            path.unlink()
        except OSError:
            pass


def _stale_reason(record, now):
    if record.get("state") == "queued":
        expires = record.get("expires")
        return f"not attached before {expires}" if expires and now > datetime.fromisoformat(expires) else None
    if not _alive(record.get("pid"), record.get("pid_started")):
        return f"process {record.get('pid')} is gone"
    return None


def sweep(announce=True):
    """Drop tickets whose process died or whose reservation expired; return the live tickets in sequence order."""
    live, now = [], _utc()
    for record in _load():
        reason = _stale_reason(record, now)
        if reason is None:
            live.append(record)
            continue
        _retire(record, reason)
        if announce:
            print(f"[queue] removed stale ticket {record['ticket']} ({record.get('owner')}: {record.get('purpose')}): {reason}", flush=True)
    return live


def _blockers(me, live):
    mine = set(me["claims"])
    return [other for other in live if other["seq"] < me["seq"] and other["ticket"] != me["ticket"] and mine & set(other["claims"])]


def _outside_holders(me):
    """Claims held without a ticket (scripts/r01_claim.py acquire) or by a ticket other than this one."""
    found = []
    for name in me["claims"]:
        current = holder(name)
        if current is not None and current.get("ticket") != me["ticket"]:
            found.append((name, current))
    return found


QUIET = COORDINATION / "queue" / "quiet.json"
DEFAULT_EXPECTED_MINUTES = 60.0


def _quiet_windows():
    """Valid windows of quiet.json as (start, end, entry) in naive local time; anything malformed is ignored."""
    try:
        windows = json.loads(QUIET.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    found = []
    for window in windows if isinstance(windows, list) else []:
        try:
            start, end = (datetime.fromisoformat(window[key]) for key in ("start", "end"))
        except (KeyError, TypeError, ValueError):
            continue
        start, end = (moment.astimezone().replace(tzinfo=None) if moment.tzinfo else moment for moment in (start, end))
        if end > start:
            found.append((start, end, window))
    return found


def _quiet_hit(me):
    """(reason, window end) when a GPU quiet window holds the ticket back now, else None.

    queue/quiet.json holds [{"start": "2026-10-14T21:00", "end": "2026-10-15T01:00", "reason": "..."}] in local time.
    A ticket does not start inside a window, nor when its expected end (--expected-minutes, default 60) would
    reach into one. A window with "scope": "timing" only holds tickets run with --timing (speed benchmarks, which
    other load on the PC would invalidate); other GPU jobs run through it. Running jobs are never interrupted.
    Delete the file or the entry to lift a window early."""
    if not me.get("claims") or "gpu" not in me["claims"]:
        return None
    timing = bool(me.get("timing"))
    now = datetime.now()
    expected = timedelta(minutes=me.get("expected_minutes") or DEFAULT_EXPECTED_MINUTES)
    for start, end, window in _quiet_windows():
        if window.get("scope", "all") == "timing" and not timing:
            continue  # the user is using the PC: accuracy jobs may run, only speed measurements would be invalid
        if now < end and now + expected > start:
            return f"GPU quiet window {window['start']} to {window['end']} ({window.get('reason', 'no reason given')})", end
    return None


def _quiet_block(me):
    """Reason the ticket must not start now because of a GPU quiet window, else None."""
    hit = _quiet_hit(me)
    return hit[0] if hit else None


def _try_start(me):
    """Take the claims and mark the ticket running. False if a claim is busy; None if the ticket was cancelled."""
    held = []
    for name in sorted(me["claims"], key=NAMES.index):
        try:
            (CLAIMS / name).mkdir(parents=True)
        except FileExistsError:
            for taken in reversed(held):
                release_claim(taken, me["owner"])
            return False
        owner = {"owner": me["owner"], "purpose": me["purpose"], "since": _iso(), "pid": os.getpid(), "ticket": me["ticket"]}
        (CLAIMS / name / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
        held.append(name)
    if not me["_path"].exists():  # cancelled between the last check and taking the claims
        for taken in reversed(held):
            release_claim(taken, me["owner"])
        return None
    me.update(state="running", started=_iso(), held=held, heartbeat=_iso())
    _write_ticket(me)
    return True


def _duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _describe(record, now=None):
    now = now or _utc()
    state = record.get("state")
    if state == "running":
        since = record.get("started") or record.get("created")
        age = _duration((now - datetime.fromisoformat(since)).total_seconds())
        return f"{record['ticket']} running {age} ({record.get('owner')}: {record.get('purpose')})"
    if state == "waiting":
        flags = (", timing" if record.get("timing") else "") + (f", expected {record['expected_minutes']:g} min" if record.get("expected_minutes") else "")
        return f"{record['ticket']} waiting{flags} ({record.get('owner')}: {record.get('purpose')})"
    return f"{record['ticket']} reserved until {record.get('expires')} ({record.get('owner')}: {record.get('purpose')})"


def _progress(me, blockers, outside, waited, live):
    parts = [f"[queue] {me['ticket']} waiting {_duration(waited)} for {', '.join(me['claims'])}"]
    if blockers:
        parts.append("behind " + "; ".join(_describe(record) for record in blockers))
    tickets = {record["ticket"] for record in live}
    outside = [(name, current) for name, current in outside if current.get("ticket") not in tickets]
    if outside:
        parts.append("claims held outside the queue: " + "; ".join(f"{name} by {current.get('owner')} since {current.get('since')} ({current.get('purpose')})" for name, current in outside))
    return ": ".join(parts[:2]) + ("; " + parts[2] if len(parts) > 2 else "")


def _cancelled_exit(me, waiting_since):
    """The ticket file is gone while this wrapper waits: it was cancelled, so leave without running the command."""
    print(f"[queue] {me['ticket']} was cancelled after waiting {_duration(time.monotonic() - waiting_since)}; the command did not run", flush=True)
    raise SystemExit(EXIT_CANCELLED)


def hold(owner, purpose, claims, heartbeat=60.0, max_wait_minutes=None, ticket=None, command=None, hold_minutes=30.0,
         expected_minutes=None, timing=False):
    """Register (or attach to) a ticket and block until it holds its claims. Returns the running ticket record."""
    _ensure_dirs()
    claims = sorted(set(claims), key=NAMES.index)
    me = _attach(ticket, owner) if ticket else _new_ticket(owner, purpose, claims, hold_minutes, command)
    me.update(state="waiting", pid=os.getpid(), pid_started=_process_started(os.getpid()), heartbeat=_iso(), command=command)
    if expected_minutes:
        me["expected_minutes"] = expected_minutes
    if timing:
        me["timing"] = True
    _write_ticket(me)
    print(f"[queue] {me['ticket']} registered for {', '.join(claims)}: {purpose}", flush=True)
    waiting_since = time.monotonic()
    deadline = None if max_wait_minutes is None else waiting_since + max_wait_minutes * 60.0
    last_line = last_beat = None
    with _Watcher(COORDINATION) as watcher:
        while True:
            live = sweep()
            if not me["_path"].exists():
                _cancelled_exit(me, waiting_since)
            # A waiting ticket that a quiet window holds back must not hold up tickets the window lets through.
            blockers = [other for other in _blockers(me, live) if not (other.get("state") == "waiting" and _quiet_block(other))]
            outside = _outside_holders(me)
            hit = None if blockers or outside else _quiet_hit(me)
            quiet = hit[0] if hit else None
            if quiet and (last_line is None or time.monotonic() - last_line >= heartbeat):
                print(f"[queue] {me['ticket']} waiting: {quiet}", flush=True)
            started = False if blockers or outside or quiet else _try_start(me)
            if started is None:  # _try_start found the ticket removed after taking the claims
                _cancelled_exit(me, waiting_since)
            if started:
                waited = time.monotonic() - waiting_since
                me["waited_seconds"] = round(waited, 1)
                print(f"[queue] {me['ticket']} started after {_duration(waited)}; holds {', '.join(me['held'])}", flush=True)
                return me
            now = time.monotonic()
            if last_line is None or now - last_line >= heartbeat:
                print(_progress(me, blockers, outside, now - waiting_since, live), flush=True)
                last_line = now
            if deadline is not None and now >= deadline:
                _retire(me, "gave up waiting")
                head = blockers[0] if blockers else None
                detail = _describe(head) if head else "; ".join(f"{name} held by {current.get('owner')} since {current.get('since')}: {current.get('purpose')}" for name, current in outside)
                print(f"[queue] {me['ticket']} gave up after {_duration(now - waiting_since)}: {detail}", flush=True)
                raise SystemExit(EXIT_WAIT_EXPIRED)
            if last_beat is None or now - last_beat >= heartbeat:
                if not me["_path"].exists():  # never recreate a cancelled ticket with a heartbeat write
                    _cancelled_exit(me, waiting_since)
                me["heartbeat"] = _iso()
                _write_ticket(me)
                last_beat = now
            timeout = heartbeat if deadline is None else max(0.05, min(heartbeat, deadline - now))
            if hit:  # wake when the window ends, not up to a heartbeat later
                timeout = max(0.05, min(timeout, (hit[1] - datetime.now()).total_seconds() + 0.2))
            watcher.wait(timeout)


def finish(record, exit_code=None, reason="finished"):
    """Release the claims a running ticket holds, record it in history.jsonl and remove the ticket."""
    for name in record.get("held") or []:
        current = holder(name)
        if current and current.get("ticket") == record["ticket"]:
            release_claim(name, current.get("owner"), force=True)
    record.update(state="finished", finished=_iso(), exit_code=exit_code, reason=reason)
    if record.get("started"):
        record["ran_seconds"] = round((_utc() - datetime.fromisoformat(record["started"])).total_seconds(), 1)
    _append_history(record)
    path = record.get("_path")
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass


def detach(record):
    """Keep the claims held (scripts/r01_claim.py acquire) but drop the ticket so the queue no longer tracks this process."""
    record.update(state="detached", finished=_iso())
    _append_history(record)
    try:
        record["_path"].unlink()
    except OSError:
        pass


def run_command(command, heartbeat=60.0, record=None):
    """Run a command bound to this process's lifetime; refresh the ticket heartbeat while it runs. Returns the exit code."""
    command = list(command[1:] if command[:1] == ["--"] else command)
    if not command:
        raise SystemExit("[queue] run needs a command after --")
    local = ROOT / command[0]
    executable = str(local) if local.is_file() else shutil.which(command[0]) or command[0]
    if executable.lower().endswith(".py"):
        command = [sys.executable, executable, *command[1:]]
    else:
        command = [executable, *command[1:]]
    process = subprocess.Popen(command, cwd=ROOT)
    job = _bind_child_lifetime(process)
    if record is not None:
        record["child_pid"] = process.pid
        _write_ticket(record)
    try:
        while True:
            try:
                return process.wait(timeout=heartbeat)
            except subprocess.TimeoutExpired:
                if record is not None:
                    record["heartbeat"] = _iso()
                    _write_ticket(record)
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        return 130
    finally:
        del job


# ----------------------------------------------------------------------------- commands

def status():
    for start, end, window in _quiet_windows():
        if end > datetime.now():
            print(f"[queue] GPU quiet window ({window.get('scope', 'all')}) {window.get('start')} to {window.get('end')}: {window.get('reason', '')}")
    _ensure_dirs()
    now = _utc()
    live = sweep()
    running = [record for record in live if record["state"] == "running"]
    waiting = [record for record in live if record["state"] != "running"]
    print("[queue] running:" if running else "[queue] running: none")
    for record in running:
        child = f", command pid {record['child_pid']}" if record.get("child_pid") else ""
        print(f"  {_describe(record, now)}; holds {', '.join(record.get('held') or [])}; wrapper pid {record.get('pid')}{child}")
    print("[queue] waiting:" if waiting else "[queue] waiting: none")
    for position, record in enumerate(waiting, 1):
        print(f"  {position}. {_describe(record, now)}; needs {', '.join(record['claims'])}")
    tickets = {record["ticket"] for record in live}
    outside = []
    for name in NAMES:
        current = holder(name)
        if current is not None and current.get("ticket") not in tickets:
            alive = "alive" if _alive(current.get("pid")) else "gone; the human may release it with r01_claim.py release --force"
            outside.append(f"  {name}: {current.get('owner')} since {current.get('since')} ({current.get('purpose')}); pid {current.get('pid')} {alive}")
    print("[queue] claims held outside the queue:" if outside else "[queue] claims held outside the queue: none")
    for line in outside:
        print(line)


def cancel(ticket, owner, force=False):
    path = TICKETS / f"{ticket}.json"
    record = _read(path)
    if record is None:
        print(f"[queue] ticket {ticket} does not exist")
        return
    if record.get("owner") != owner and not force:
        raise SystemExit(f"[queue] ticket {ticket} belongs to {record.get('owner')}, not {owner}; only the human may pass --force")
    record["_path"] = path
    if record.get("state") == "running" and _alive(record.get("pid"), record.get("pid_started")):
        print(f"[queue] note: {ticket} is running under pid {record.get('pid')}; cancelling releases its claims but does not stop the command")
    waiting_pid = None
    if record.get("state") in ("queued", "waiting") and _alive(record.get("pid"), record.get("pid_started")):
        waiting_pid = record.get("pid")
    _retire(record, f"cancelled by {owner}")
    print(f"[queue] cancelled {ticket}")
    if waiting_pid:
        # The wrapper notices the missing ticket on its next wake-up (directory-change notification) and exits with
        # EXIT_CANCELLED. If it is still alive after the grace period (it may have been starting at this moment),
        # terminate it; its command, if any, is bound to the wrapper's lifetime and ends with it.
        deadline = time.monotonic() + CANCEL_GRACE_SECONDS
        while time.monotonic() < deadline and _alive(waiting_pid, record.get("pid_started")):
            time.sleep(0.1)
        if _alive(waiting_pid, record.get("pid_started")):
            import psutil

            try:
                psutil.Process(waiting_pid).terminate()
                print(f"[queue] terminated waiting wrapper pid {waiting_pid} of {ticket}")
            except psutil.Error as error:
                print(f"[queue] warning: could not terminate waiting wrapper pid {waiting_pid} of {ticket}: {error}")
        else:
            print(f"[queue] waiting wrapper pid {waiting_pid} of {ticket} exited without running its command")


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("run", "enqueue"):
        p = sub.add_parser(action)
        p.add_argument("--owner", required=True, help="session name, as in docs/R01_PARALLEL_SESSIONS.md")
        p.add_argument("--purpose", required=True)
        p.add_argument("--claim", action="append", required=True, choices=NAMES)
        p.add_argument("--hold-minutes", type=float, default=30.0, help="enqueue: minutes a reservation stays valid before `run --ticket` attaches")
        if action == "run":
            p.add_argument("--ticket", help="attach to a reservation made with enqueue (keeps its place in the queue)")
            p.add_argument("--heartbeat-seconds", type=float, default=60.0)
            p.add_argument("--expected-minutes", type=float, help="expected run time; a ticket never starts if it would run into a GPU quiet window (default 60)")
            p.add_argument("--timing", action="store_true", help="speed measurement: held by quiet windows with scope timing")
            p.add_argument("--max-wait-minutes", type=float, help="give up with exit code 75 instead of waiting indefinitely")
            p.add_argument("command", nargs=REMAINDER)
    p = sub.add_parser("cancel")
    p.add_argument("--ticket", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--force", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "status":
        status()
        return
    if args.action == "cancel":
        cancel(args.ticket, args.owner, args.force)
        return
    claims = sorted(set(args.claim), key=NAMES.index)
    if args.action == "enqueue":
        record = _new_ticket(args.owner, args.purpose, claims, args.hold_minutes)
        print(record["ticket"])
        print(f"[queue] reserved {record['ticket']} for {', '.join(claims)}; attach with `run --ticket {record['ticket']}` before {record['expires']}", flush=True)
        return
    heartbeat = min(max(args.heartbeat_seconds, 30.0), 600.0)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("run needs a command after --")
    record = hold(args.owner, args.purpose, claims, heartbeat, args.max_wait_minutes, args.ticket, command, args.hold_minutes,
                  args.expected_minutes, args.timing)
    code = None
    try:
        code = run_command(command, heartbeat, record)
    finally:
        finish(record, code)
        print(f"[queue] {record['ticket']} finished with exit code {code} after {_duration(record.get('ran_seconds') or 0)}; released {', '.join(record.get('held') or [])}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
