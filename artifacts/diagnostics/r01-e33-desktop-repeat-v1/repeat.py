"""One human-authorized historical E33 repeat; no optimization selection."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.r01_workflow import configure_runtime, run_logged, write_json
from scripts.gpu_r01_policy import result_metadata, launch_plan
from scripts.r01_iterate import load_snapshot
from argparse import Namespace

R = Path('artifacts/retraining/gpu-retrain-r01')
OUT = ROOT / R / 'iteration-0040'
TRACK = 'E30-F37-F42'
RUNTIME = R / 'runtime-0033-b2cc642dbf2e'
REASON = 'Human requested one repeat of iteration 0033 while actively using the desktop: "pop a worker to re-run iteration 33 to see if we still achieve the same results with me using the dekstop". Changed background desktop workload makes historical timing insufficient for this reproducibility diagnostic.'

if sys.argv[1] == 'prepare':
    if OUT.exists():
        raise FileExistsError('Fresh evidence directory required')
    original = ROOT / R / 'iteration-0033/protocol.json'
    protocol = json.loads(original.read_text(encoding='utf-8'))
    archive = ROOT / protocol['preparation_provenance']['source_archive']
    with ZipFile(archive) as z:
        manifest = json.loads(z.read('source-manifest.json'))
        mismatches = [name for name, digest in manifest['files'].items()
                      if hashlib.sha256((ROOT / RUNTIME / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError(f'Historical source differs: {mismatches}')
    reference = ROOT / R / f'iteration-0033/expertium/{TRACK}/result.json'
    protocol.update(iteration='0040', authorization=REASON, round_authorization=REASON,
                    round_start_iteration='0040', prior_no_20_percent_gain_streak={'expertium': 0},
                    control_tracks={'expertium': TRACK}, candidate_tracks={'expertium': []},
                    case_options_by_model={'expertium': {TRACK: protocol['case_options_by_model']['expertium'][TRACK]}},
                    resource_retuning_pending={'expertium': False}, retuning_candidates={},
                    models=['expertium'], paused_models=['baseline'], previous_iteration='0033',
                    references={'expertium': {'previous': result_metadata(reference, ROOT)}},
                    diagnostic_repeat={'scope': 'one full matched historical-source repeat; no optimization round or broader validation',
                                       'reason': REASON, 'original_protocol': result_metadata(original, ROOT),
                                       'source_files_verified': len(manifest['files']),
                                       'historical_runtime': RUNTIME.as_posix(),
                                       'optimization_stop_preserved': '0039 remains three misses; this authorization covers this repeat only'},
                    frozen_utc=datetime.now(timezone.utc).isoformat())
    loop = (ROOT / 'docs/GPU_RETRAIN_LOOP.md').read_bytes()
    protocol['loop_sha256'] = hashlib.sha256(loop).hexdigest()
    OUT.mkdir(parents=True)
    (OUT / 'loop-definition.md').write_bytes(loop)
    write_json(OUT / 'protocol.json', protocol)
    write_json(OUT / 'repeat-compatibility.json', {
        'source_archive': archive.relative_to(ROOT).as_posix(), 'source_files_verified': len(manifest['files']),
        'training_source_and_options': 'unchanged from original 0033',
        'reused_arithmetic_and_smoke_evidence': reference.relative_to(ROOT).as_posix(),
        'preflight': 'artifacts/diagnostics/r01-e33-desktop-repeat-v1/preflight.json',
        'new_smoke_required': False, 'runtime': RUNTIME.as_posix(), 'reason': REASON})
    args = Namespace(model='expertium', track=TRACK, smoke=False, repeat_reason=REASON, check_reason=None)
    print(json.dumps(launch_plan(ROOT, OUT, args)))
elif sys.argv[1] == 'run':
    configure_runtime(ROOT, 1)
    load = load_snapshot()
    import subprocess
    load['gpu_state'] = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=5).stdout.strip()
    load['recorded_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(OUT / 'background-load.json', load)
    print('[repeat] Background before timing: ' + json.dumps(load), flush=True)
    command = [sys.executable, '-X', 'utf8', str(ROOT / 'scripts/run_isolated_gpu_r01.py'), '--case',
               '--runtime', str(RUNTIME), '--output', str(OUT.relative_to(ROOT)), '--model', 'expertium',
               '--track', TRACK, '--repeat-reason', REASON]
    run_logged('desktop-repeat', command, OUT / 'console.log', root=ROOT, heartbeat_seconds=60, timeout_seconds=600)
    print('[repeat] Completed one full historical-source run.', flush=True)
