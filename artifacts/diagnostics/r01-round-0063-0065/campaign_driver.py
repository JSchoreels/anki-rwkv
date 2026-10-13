import json, subprocess, sys
from pathlib import Path
R=Path('artifacts/retraining/gpu-retrain-r01')
D=Path('artifacts/diagnostics/r01-round-0063-0065')
py=sys.executable
def run(args):
 print('[driver] '+ ' '.join(args),flush=True)
 subprocess.run([py,*args],check=True)
def gate(n):
 out=R/f'iteration-{n}'
 run(['scripts/report_gpu_r01_next.py','--output',str(out),'--brief'])
 while True:
  c=json.loads((out/'comparison.json').read_text())['comparisons']['expertium']
  b=c.get('replicated_blocks') or {}
  print('[gate] '+json.dumps({'iteration':n,'blocks':b.get('n'),'mean':b.get('mean_percent'),'lower':b.get('lower_bound_percent'),'decision':b.get('decision'),'final':b.get('final')}),flush=True)
  if b.get('final'): return
  plan=json.loads((R/'plans'/f'iteration-{n}-codex.json').read_text())
  a=plan['control_tracks']['expertium']; candidate=plan['candidate_tracks']['expertium']
  run(['scripts/r01_replicates.py','run','--output',str(out),'--sequence',','.join([a,candidate]*2),'--reason','Complete next frozen replicated-block look for changed exact recurrence schedule'])
  run(['scripts/report_gpu_r01_next.py','--output',str(out),'--brief'])
if len(sys.argv)>1:
 gate(sys.argv[1])
 sys.exit(0)
# 0063 finalized; 0064 completed training/profile and resumed report explicitly.
for n,prev,mode,name,track,explanation in [
 ('0064','0063',7,'long','E33-F48-F50-F58g','Select forward scalar/value prefetch only for histories at least 32 steps, preserving baseline elsewhere; test sequential latency hiding versus live-set pressure.'),
 ('0065','0064',8,'short','E33-F48-F50-F58h','Select forward scalar/value prefetch only for histories at most 16 steps, preserving baseline elsewhere; test broad short-history scheduling separately from long chains.')]:
 if n == '0064':
  gate(n)
  continue
 run(['scripts/r01_iterate.py','--iteration',n,'--previous',prev,'--agent','codex','--candidate',track,'--set',f'recurrence_load_pipeline={mode}','--set','recurrence_prefetch=false','--explanation',explanation,'--evidence','Nine real WKV shapes distinguish long chains from broad short histories; 216 synthetic cases and five real loss/every-gradient checks on final source passed.','--check-reason','Changed shape-selected forward load schedule','--preflight',str(D/'preflight.json'),'--check',str(D/f'synthetic-{name}/result.json'),'--check',str(D/f'real-{name}/result.json')])
 gate(n)
