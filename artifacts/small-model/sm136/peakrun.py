"""Run a command, poll its working set every 2 s (children included), print the peak GB and elapsed seconds."""
import subprocess, sys, time, psutil
t0 = time.time(); p = subprocess.Popen(sys.argv[1:]); ps = psutil.Process(p.pid); peak = 0; trace = []
while p.poll() is None:
    try:
        rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
    except psutil.Error:
        rss = 0
    peak = max(peak, rss); trace.append(round(rss / 2**30, 2)); time.sleep(2)
print(f"PEAK {peak/2**30:.2f} GB rc={p.returncode} {time.time()-t0:.0f}s trace(every 20s)={trace[::10]}", flush=True)
