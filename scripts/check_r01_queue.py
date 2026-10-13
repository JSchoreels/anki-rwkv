"""CPU-only checks for scripts/r01_queue.py and the queued scripts/r01_claim.py (no GPU, no training).

Spawns real wrapper processes against a private coordination root under the output directory and verifies FIFO order,
concurrency of disjoint claims, event-driven wake-up (well under the 30 s heartbeat), stale-ticket removal, claims held
outside the queue, reservations, give-up exit codes, cancellation, and cancellation of a waiting wrapper (it must exit without running).
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance  # noqa: E402

PYTHON = sys.executable
QUEUE = str(ROOT / "scripts" / "r01_queue.py")
CLAIM = str(ROOT / "scripts" / "r01_claim.py")
CHILD = (
    "import sys, time\n"
    "log, name, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])\n"
    "open(log, 'a').write(f'{name} start {time.time()}\\n')\n"
    "time.sleep(seconds)\n"
    "open(log, 'a').write(f'{name} end {time.time()}\\n')\n"
)
WAKE_LIMIT = 2.5  # seconds between a release and the next start; the heartbeat is 30 s, so polling would fail this


class Harness:
    def __init__(self, output):
        self.output = output
        self.root = output / "coordination"
        self.log = output / "children.log"
        self.env = {**os.environ, "R01_COORDINATION_ROOT": str(self.root), "PYTHONIOENCODING": "utf-8"}
        self.processes = {}
        self.results = {}

    def run(self, name, seconds, claims, extra=(), tool=QUEUE, wait_minutes=None):
        if tool == QUEUE:
            command = [PYTHON, QUEUE, "run", "--owner", name, "--purpose", f"test {name}", "--heartbeat-seconds", "30", *extra]
        else:
            command = [PYTHON, CLAIM, "run", "--owner", name, "--purpose", f"test {name}", "--wait-minutes", str(wait_minutes), *extra]
        for claim in claims:
            command += ["--claim", claim]
        command += ["--", PYTHON, "-c", CHILD, str(self.log), name, str(seconds)]
        process = subprocess.Popen(command, cwd=ROOT, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        self.processes[name] = process
        return process

    def wait(self, *names, timeout=30):
        for name in names:
            process = self.processes[name]
            output, _ = process.communicate(timeout=timeout)
            self.results[name] = {"exit_code": process.returncode, "output": output.strip().splitlines()[-6:]}

    def events(self):
        marks = {}
        if self.log.exists():
            for line in self.log.read_text(encoding="utf-8").splitlines():
                name, kind, stamp = line.split()
                marks[(name, kind)] = float(stamp)
        return marks

    def wait_for(self, name, kind, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            stamp = self.events().get((name, kind))
            if stamp is not None:
                return stamp
            time.sleep(0.05)
        raise AssertionError(f"{name} {kind} not observed within {timeout} s")

    def command(self, *arguments, tool=QUEUE, check=True):
        completed = subprocess.run([PYTHON, tool, *arguments], cwd=ROOT, env=self.env, capture_output=True, text=True, encoding="utf-8")
        if check and completed.returncode != 0:
            raise AssertionError(f"{arguments} failed: {completed.stdout}\n{completed.stderr}")
        return completed

    def tickets(self):
        return sorted(path.name for path in (self.root / "queue" / "tickets").glob("*.json"))


def check_fifo_and_wake(h, report):
    h.run("A", 3, ["gpu"])
    h.wait_for("A", "start")
    h.run("B", 0.5, ["gpu"])
    time.sleep(0.7)
    h.run("C", 0.2, ["gpu"])
    h.run("D", 0.2, ["ledger"])  # disjoint claim: must not wait for A
    h.wait("A", "B", "C", "D")
    e = h.events()
    assert e[("D", "start")] < e[("A", "end")], "disjoint claims should run concurrently"
    assert e[("A", "end")] <= e[("B", "start")] < e[("B", "end")] <= e[("C", "start")], "gpu tickets must run one at a time in order"
    wake_b, wake_c = e[("B", "start")] - e[("A", "end")], e[("C", "start")] - e[("B", "end")]
    assert wake_b < WAKE_LIMIT and wake_c < WAKE_LIMIT, (wake_b, wake_c)
    assert all(h.results[n]["exit_code"] == 0 for n in "ABCD"), h.results
    report["fifo"] = {"wake_after_release_s": [round(wake_b, 3), round(wake_c, 3)], "overlap": False}


def check_outside_holder(h, report):
    h.command("acquire", "--claim", "gpu", "--owner", "human", "--purpose", "manual hold", tool=CLAIM)
    h.run("E", 0.2, ["gpu"])
    time.sleep(1.5)
    assert ("E", "start") not in h.events(), "a claim held outside the queue must block"
    released = time.time()
    h.command("release", "--claim", "gpu", "--owner", "human", tool=CLAIM)
    h.wait("E")
    wake = h.events()[("E", "start")] - released
    assert 0 <= wake < WAKE_LIMIT, wake
    report["outside_holder"] = {"wake_after_release_s": round(wake, 3)}


def check_stale_ticket(h, report):
    import psutil

    ghost = subprocess.Popen([PYTHON, "-c", "import time; time.sleep(0.4)"])
    started = psutil.Process(ghost.pid).create_time()
    ghost.wait()
    tickets = h.root / "queue" / "tickets"
    claim = h.root / "claims" / "gpu"
    claim.mkdir(parents=True)
    (claim / "owner.json").write_text(json.dumps({"owner": "ghost", "purpose": "crashed", "since": "2026-01-01T00:00:00+00:00", "pid": ghost.pid, "ticket": "000001-ghost"}), encoding="utf-8")
    (tickets / "000001-ghost.json").write_text(json.dumps({"ticket": "000001-ghost", "seq": 1, "owner": "ghost", "purpose": "crashed", "claims": ["gpu"], "state": "running", "created": "2026-01-01T00:00:00+00:00", "started": "2026-01-01T00:00:00+00:00", "pid": ghost.pid, "pid_started": started, "held": ["gpu"]}), encoding="utf-8")
    launched = time.time()
    h.run("F", 0.2, ["gpu"])
    h.wait("F")
    assert h.results["F"]["exit_code"] == 0, h.results["F"]
    assert (h.root / "queue" / "stale" / "000001-ghost.json").exists(), "stale ticket should be kept under stale/"
    assert not claim.exists(), "the dead ticket's claim should have been released"
    delay = h.events()[("F", "start")] - launched
    report["stale"] = {"start_after_launch_s": round(delay, 3), "removed_line": [line for line in h.results["F"]["output"] if "stale" in line][:1]}


def check_reservation_and_cancel(h, report):
    reserved = h.command("enqueue", "--owner", "G", "--purpose", "reserve", "--claim", "gpu").stdout.splitlines()[0].strip()
    h.run("H", 0.2, ["gpu"])
    time.sleep(1.0)
    assert ("H", "start") not in h.events(), "a reservation with a lower sequence must block later tickets"
    h.run("G", 0.3, ["gpu"], extra=["--ticket", reserved])
    h.wait("G", "H")
    e = h.events()
    assert e[("G", "start")] < e[("H", "start")] and e[("G", "end")] <= e[("H", "start")], "the reservation must run first"
    cancelled = h.command("enqueue", "--owner", "J", "--purpose", "cancel me", "--claim", "ledger").stdout.splitlines()[0].strip()
    assert f"{cancelled}.json" in h.tickets()
    refused = h.command("cancel", "--ticket", cancelled, "--owner", "someone-else", check=False)
    assert refused.returncode != 0, "another owner must not cancel a ticket without --force"
    h.command("cancel", "--ticket", cancelled, "--owner", "J")
    assert f"{cancelled}.json" not in h.tickets()
    report["reservation"] = {"ticket": reserved, "ran_before_later_ticket": True, "cancel_refused_for_other_owner": True}


def check_cancel_waiting_wrapper(h, report):
    """Regression (2026-10-13): cancelling a ticket whose wrapper is waiting must stop it; the command must never run."""
    h.run("L", 3, ["gpu"])
    h.wait_for("L", "start")
    h.run("M", 0.2, ["gpu"])
    deadline = time.monotonic() + 10
    while not any(name.endswith("-M.json") for name in h.tickets()) and time.monotonic() < deadline:
        time.sleep(0.05)
    ticket = next(name for name in h.tickets() if name.endswith("-M.json"))[:-5]
    time.sleep(0.5)  # let M settle into its wait
    cancelled_at = time.time()
    output = h.command("cancel", "--ticket", ticket, "--owner", "M").stdout
    h.wait("M", timeout=15)
    exited = time.time() - cancelled_at
    assert h.results["M"]["exit_code"] == 76, h.results["M"]
    h.wait("L")
    time.sleep(1.0)  # a surviving wrapper would start right after L ends
    assert ("M", "start") not in h.events(), "a cancelled waiting ticket must never run its command"
    assert not any(name.endswith("-M.json") for name in h.tickets()), "a cancelled ticket must not be recreated"
    assert not (h.root / "claims" / "gpu").exists(), "no claim may stay held"
    report["cancel_waiting"] = {"wrapper_exit_code": 76, "wrapper_exit_after_cancel_s": round(exited, 3),
                                "cancel_output": output.strip().splitlines()}


def check_give_up_and_claim_run(h, report):
    h.run("A2", 3, ["gpu"])
    h.wait_for("A2", "start")
    h.run("I", 0.2, ["gpu"], extra=["--max-wait-minutes", "0.02"])
    h.run("K", 0.2, ["gpu"], tool=CLAIM, wait_minutes=1)
    h.wait("I", timeout=10)
    assert h.results["I"]["exit_code"] == 75, h.results["I"]
    assert ("I", "start") not in h.events()
    h.wait("A2", "K")
    e = h.events()
    wake = e[("K", "start")] - e[("A2", "end")]
    assert 0 <= wake < WAKE_LIMIT, wake
    assert h.results["K"]["exit_code"] == 0, h.results["K"]
    assert not (h.root / "claims" / "gpu").exists() and h.tickets() == [], (h.tickets(), list((h.root / "claims").iterdir()))
    report["give_up"] = {"exit_code": 75, "claim_run_wake_s": round(wake, 3)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    output.mkdir(parents=True)
    h = Harness(output)
    report = {"provenance": process_provenance(), "wake_limit_s": WAKE_LIMIT}
    started = time.monotonic()
    for check in (check_fifo_and_wake, check_outside_holder, check_stale_ticket, check_reservation_and_cancel, check_cancel_waiting_wrapper, check_give_up_and_claim_run):
        print(f"[check] {check.__name__}", flush=True)
        check(h, report)
    status = h.command("status").stdout
    history = (h.root / "queue" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    report.update(status=status.splitlines(), history_entries=len(history), wrapper_results=h.results, elapsed_s=round(time.monotonic() - started, 1))
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[check] all queue checks passed in {report['elapsed_s']} s; {len(history)} history entries; result {output / 'result.json'}")


if __name__ == "__main__":
    main()
