"""Diagnostic: sample RSS every 20 ms with the running module stack; report where the RSS maximum occurs."""
import sys, threading, time, psutil, torch, runpy
sys.path.insert(0, 'C:/Users/jonat/workspace/wt/sm136')
proc = psutil.Process(); stack = []; samples = []; stop = [False]
def pre(m, a): stack.append(type(m).__name__)
def post(m, a, o):
    if stack: stack.pop()
torch.nn.modules.module.register_module_forward_pre_hook(pre)
torch.nn.modules.module.register_module_forward_hook(post)
def sampler():
    while not stop[0]:
        samples.append((proc.memory_info().rss, '/'.join(stack[-4:]), time.time())); time.sleep(0.02)
threading.Thread(target=sampler, daemon=True).start()
sys.argv = ['x'] + sys.argv[1:]
try: runpy.run_path('C:/Users/jonat/workspace/wt/sm136/scripts/small_model_heldout.py', run_name='__main__')
except SystemExit: pass
stop[0] = True
best = {}
for r, s, t in samples:
    best[s] = max(best.get(s, 0), r)
print('MAX', max(r for r, _, _ in samples) / 2**30)
for s, r in sorted(best.items(), key=lambda x: -x[1])[:12]: print(round(r / 2**30, 2), s or '(outside modules)')
# first sample at which RSS passes 90% of max
mx = max(r for r, _, _ in samples)
for r, s, t in samples:
    if r > 0.9 * mx: print('first >90%:', round(r / 2**30, 2), s or '(outside modules)', round(t - samples[0][2], 1), 's of', round(samples[-1][2] - samples[0][2]), 's'); break
t0 = samples[0][2]; last = -1
for r, s, t in samples:
    if int(t - t0) // 3 != last:
        last = int(t - t0) // 3; print(f't={t-t0:5.1f}s rss={r/2**30:.2f} {s or "-"}')
