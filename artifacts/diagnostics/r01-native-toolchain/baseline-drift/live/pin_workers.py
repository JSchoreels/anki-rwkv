"""External diagnostic: pin R01 preparation workers one per physical core (no training-source change).

Watches for a python process that owns >= 6 `--multiprocessing-fork` children (the prestarted
preparation pool) and sets each worker's OS affinity to the first logical CPU of a distinct physical
core, leaving the remaining cores and all SMT siblings free for the main/driver threads. Writes what
it pinned to --log and exits after the first pool is pinned or --timeout seconds.

  python pin_workers.py --log <json> [--timeout 600] [--workers 6]
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import ctypes
import json
import struct
import time
from ctypes import wintypes

import psutil


def physical_cores():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    size = wintypes.ULONG(0)
    k.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = ctypes.create_string_buffer(size.value)
    if not k.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_last_error(), "GetSystemCpuSetInformation failed")
    cores, off = {}, 0
    while off < size.value:
        length, = struct.unpack_from("<I", buf.raw, off)
        _, _, cpu, core, _, _, _, _, rank = struct.unpack_from("<IHBBBBBBB", buf.raw, off + 8)
        cores.setdefault(core, {"cpus": [], "rank": rank})["cpus"].append(cpu)
        off += length
    return cores


def fork_children(parent):
    out = []
    for child in parent.children():
        try:
            if child.name().lower().startswith("python") and "--multiprocessing-fork" in " ".join(child.cmdline()):
                out.append(child)
        except psutil.Error:
            pass
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--main-cpus", default="", help="Optional comma list: also confine the training (pool parent) process")
    args = parser.parse_args()
    cores = physical_cores()
    # Workers get core indices 2,4,...: physical cores 0 and the last stay entirely free for the main process.
    order = sorted(cores)
    targets = [cores[c]["cpus"][0] for c in order[1:1 + args.workers]]
    started = time.monotonic()
    while time.monotonic() - started < args.timeout:
        for proc in psutil.process_iter(["name"]):
            if not (proc.info["name"] or "").lower().startswith("python"):
                continue
            try:
                kids = fork_children(proc)
            except psutil.Error:
                continue
            if len(kids) >= args.workers:
                pins = []
                for kid, cpu in zip(sorted(kids, key=lambda p: p.pid), targets):
                    kid.cpu_affinity([cpu])
                    pins.append({"pid": kid.pid, "cpu": cpu, "affinity": kid.cpu_affinity()})
                main_affinity = None
                if args.main_cpus:
                    proc.cpu_affinity([int(c) for c in args.main_cpus.split(",")])
                    main_affinity = proc.cpu_affinity()
                record = {"main_affinity": main_affinity,"recorded_utc": datetime.now(timezone.utc).isoformat(), "parent_pid": proc.pid,
                          "parent_cmdline": " ".join(proc.cmdline())[:400], "workers_found": len(kids),
                          "pins": pins, "seconds_after_start": round(time.monotonic() - started, 2),
                          "topology": {str(c): v for c, v in sorted(cores.items())}}
                with open(args.log, "w", encoding="utf-8") as fh:
                    json.dump(record, fh, indent=2)
                print(json.dumps({"pinned": [p["cpu"] for p in pins], "parent": proc.pid}), flush=True)
                return
        time.sleep(0.25)
    raise SystemExit("no preparation pool found before timeout")


if __name__ == "__main__":
    main()
