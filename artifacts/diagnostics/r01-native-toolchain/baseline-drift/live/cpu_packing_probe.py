"""CPU-only probe: do 6 busy worker processes get packed onto SMT siblings?"""
import ctypes, struct, sys, time, multiprocessing as mp
from ctypes import wintypes
import psutil


def topology():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    size = wintypes.ULONG(0)
    k.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = ctypes.create_string_buffer(size.value)
    k.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0)
    cores, off = {}, 0
    while off < size.value:
        length, = struct.unpack_from("<I", buf.raw, off)
        _, _, cpu, core, _, _, _, _, rank = struct.unpack_from("<IHBBBBBBB", buf.raw, off + 8)
        cores.setdefault(core, {"cpus": [], "rank": rank})["cpus"].append(cpu)
        off += length
    return cores


def work(seconds, q, cpus):
    if cpus:
        psutil.Process().cpu_affinity(cpus)
    import numpy as np
    a = np.random.default_rng(0).random(200_000)
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        # mix of interpreter work and small numpy ops, like the preparation workers
        s = 0
        for i in range(2000):
            s += i * i
        np.sort(a[:20000])
        n += 1
    q.put(n)


def run(label, pins, seconds=12):
    q = mp.Queue()
    ps = [mp.Process(target=work, args=(seconds, q, pins[i] if pins else None)) for i in range(6)]
    for p in ps:
        p.start()
    time.sleep(1.5)
    psutil.cpu_percent(percpu=True)
    samples = []
    for _ in range(8):
        time.sleep(1)
        samples.append(psutil.cpu_percent(percpu=True))
    for p in ps:
        p.join()
    counts = [q.get() for _ in ps]
    per = [sum(c) / len(samples) for c in zip(*samples)]
    print(f"{label:28s} iterations/proc mean={sum(counts)/6:8.0f} min={min(counts):6d} max={max(counts):6d}")
    print("   per-logical-cpu %:", " ".join(f"{x:3.0f}" for x in per))
    return sum(counts) / 6


if __name__ == "__main__":
    cores = topology()
    print("physical cores -> logical cpus (rank):", {c: (v["cpus"], v["rank"]) for c, v in sorted(cores.items())})
    order = sorted(cores.items(), key=lambda kv: -kv[1]["rank"])
    # leave the best-ranked core for the main thread, give each worker its own physical core
    spread = [[v["cpus"][0]] for _, v in order[1:7]]
    packed = []
    for _, v in order[1:4]:
        packed += [[v["cpus"][0]], [v["cpus"][1]]]
    a = run("default scheduler", None)
    b = run("pinned 1 per physical core", spread)
    c = run("pinned 2 per core (SMT)", packed)
    d = run("default scheduler again", None)
    print(f"ratios vs spread: default={a/b:.3f} smt-packed={c/b:.3f} default2={d/b:.3f}")
