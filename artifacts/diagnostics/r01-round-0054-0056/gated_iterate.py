"""Current workspace iterator with the existing replicate quiet gate at full launch."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import r01_iterate
from scripts.r01_replicates import wait_for_quiet
original = r01_iterate.load_snapshot
def gated_snapshot():
    quiet = wait_for_quiet(10.0, 1200)
    iteration = sys.argv[sys.argv.index('--iteration')+1]
    (Path(__file__).parent/f'quiet-{iteration}.json').write_text(json.dumps(quiet, indent=2)+'\n')
    return original()
r01_iterate.load_snapshot = gated_snapshot
r01_iterate.main()
