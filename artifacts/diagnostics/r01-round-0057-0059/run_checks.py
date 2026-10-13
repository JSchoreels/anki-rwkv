import subprocess, sys
base='artifacts/diagnostics/r01-round-0057-0059'
for mode in (1,2,3):
 if mode != 1:
  subprocess.run([sys.executable,'scripts/r01_preflight.py','--output',base+'/preflight.json','--run','scripts/check_r01_compact_previous.py','--prefetch','--load-pipeline',str(mode),'--output',base+f'/synthetic-{mode}'],check=True)
 subprocess.run([sys.executable,'scripts/r01_preflight.py','--output',base+'/preflight.json','--run','scripts/check_r01_exact_options.py','--a',base+'/accepted-options.json','--b',base+f'/pipeline-{mode}-options.json','--output',base+f'/real-{mode}-final'],check=True)

