"""F67 fail-closed evaluation reuse proof and the runtime manifest it relies on (F74 T1).

Training records `evaluation_context` (runtime manifest, NVRTC, determinism, evaluation source) in every
evaluated case. An exact replica may then skip held-out evaluation only when `prove_reuse` accepts the origin:
same final parameters (re-hashed from both checkpoints), same prepared held-out input hash, same options,
config, implementation, protocol context and complete runtime manifest. Any missing or unequal field refuses.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPORT_FIELDS = ('model', 'torch', 'cuda', 'gpu', 'master_dtype', 'float32_matmul_precision',
                 'cuda_matmul_allow_tf32', 'implementation_sha256', 'trained_parameter_sha256',
                 'validation_input_sha256', 'evaluation_users', 'config', 'evaluation_context', 'options')
PROTOCOL_FIELDS = ('dataset_revision', 'cohort_sha256', 'evaluation_rows', 'evaluation', 'holdout', 'seeds')
CONTEXT_FIELDS = ('driver_version', 'nvrtc_version', 'backend', 'deterministic_algorithms',
                  'cublas_workspace_config', 'evaluation_dtype', 'prepared_input_sha256', 'evaluation_source_sha256', 'runtime')
# Everything that identifies the numerical runtime of a recorded run and can be read again before a launch.
RUNTIME_FIELDS = ('torch', 'numpy', 'cupy', 'nvrtc_package', 'cuda_path', 'driver_version', 'gpu_name')
EVALUATION_SOURCES = ('scripts/compare.py', 'scripts/gpu_batch1.py', 'scripts/run_gpu_batch1.py', 'scripts/gpu_r01_training.py',
                      'scripts/r01_evaluation_reuse.py', 'rwkv/prepare_batch.py')
ORIGIN_ENVIRONMENT = 'GPU_R01_EVALUATION_ORIGIN'


def runtime_manifest():
    """Package builds, CUDA toolkit path and driver; same fields as the preflight runtime record."""
    from importlib import metadata
    def version(*names):
        for name in names:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        return None
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=10)
        driver, _, gpu = query.stdout.strip().partition(',') if query.returncode == 0 else ('', '', '')
    except (OSError, subprocess.TimeoutExpired):
        driver, gpu = '', ''
    return {'torch': version('torch'), 'numpy': version('numpy'), 'cupy': version('cupy-cuda13x', 'cupy-cuda12x', 'cupy'),
            'nvrtc_package': version('nvidia-cuda-nvrtc-cu13', 'nvidia-cuda-nvrtc-cu12') or 'none',
            'cuda_path': os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME') or 'none',
            'driver_version': driver.strip() or None, 'gpu_name': gpu.strip() or None}


def evaluation_source_sha256(root, provenance):
    """Digest of the frozen evaluation-path sources (rwkv/, expertium/ and the evaluation scripts)."""
    from zipfile import ZipFile
    with ZipFile(Path(root) / provenance['source_archive']) as archive:
        files = json.loads(archive.read('source-manifest.json'))['files']
    chosen = {name: digest for name, digest in files.items()
              if name in EVALUATION_SOURCES or name.startswith(('rwkv/', 'expertium/'))}
    return hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()


def evaluation_context(root, report, provenance):
    """Manifest recorded after training; outside the training timer. Needs the prepared-input hash first."""
    import torch
    try:
        import cupy
        nvrtc = '.'.join(str(part) for part in cupy.cuda.nvrtc.getVersion())
    except Exception:  # noqa: BLE001 - a missing NVRTC version leaves the manifest incomplete, which refuses reuse
        nvrtc = None
    config = report.get('config') or {}
    return {'driver_version': runtime_manifest()['driver_version'], 'nvrtc_version': nvrtc,
            'backend': json.dumps({'wkv_backend': config.get('WKV_BACKEND'), 'recurrence_kernel': report.get('recurrence_kernel')}, sort_keys=True),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'evaluation_dtype': config.get('DTYPE'), 'prepared_input_sha256': report.get('validation_input_sha256'),
            'evaluation_source_sha256': evaluation_source_sha256(root, provenance), 'runtime': runtime_manifest()}


def complete_runtime(runtime):
    return isinstance(runtime, dict) and all(runtime.get(field) not in (None, '', 'unavailable') for field in RUNTIME_FIELDS)


def prove_reuse(candidate_report, candidate_protocol, full_directory, full_protocol, *, candidate_checkpoint=None):
    """Caller must independently hash the final checkpoint and prepared evaluation inputs first."""
    full_directory = Path(full_directory)
    result_path = full_directory / 'result.json'
    raw = result_path.read_bytes()
    full = json.loads(raw)
    failures = []
    if candidate_checkpoint is None or not Path(candidate_checkpoint).is_file() or not (full_directory / 'master.pth').is_file():
        failures.append('Missing final checkpoint proof')
    else:
        import torch
        from scripts.run_gpu_batch1 import tensor_hash
        for checkpoint, report in ((Path(candidate_checkpoint), candidate_report), (full_directory / 'master.pth', full)):
            actual = tensor_hash(torch.load(checkpoint, map_location='cpu', weights_only=True))
            if actual != report.get('trained_parameter_sha256'):
                failures.append('Final checkpoint contents disagree with recorded parameter hash')
    if full.get('status') != 'complete' or not isinstance(full.get('smoke_only'), bool) \
            or full.get('smoke_only') is not candidate_report.get('smoke_only'):
        failures.append('Evaluation origin must be a completed case of the same scope (full or smoke) as the candidate')
    if (full.get('evaluation_reuse') or {}).get('status') == 'reused':
        failures.append('Origin must have evaluated itself; chained reuse refused')
    for field in REPORT_FIELDS:
        if field not in candidate_report or field not in full or candidate_report[field] in (None, '') or candidate_report[field] != full[field]:
            failures.append(f'Missing or unequal report proof: {field}')
    for report in (candidate_report, full):
        context = report.get('evaluation_context')
        if not isinstance(context, dict) or any(field not in context or context[field] in (None, '') for field in CONTEXT_FIELDS):
            failures.append('Missing complete evaluation runtime/source manifest')
        elif context['prepared_input_sha256'] != report.get('validation_input_sha256') or not context['deterministic_algorithms']:
            failures.append('Evaluation manifest lacks deterministic exact input proof')
        elif not complete_runtime(context['runtime']):
            failures.append('Incomplete runtime manifest')
    for field in PROTOCOL_FIELDS:
        if field not in candidate_protocol or field not in full_protocol or candidate_protocol[field] != full_protocol[field]:
            failures.append(f'Missing or unequal protocol proof: {field}')
    if not full.get('per_user') or not full.get('evaluation'):
        failures.append('Full origin lacks per-user losses or aggregate evaluation')
    if failures:
        return {'status':'refused', 'reasons':failures}
    return {'status':'reusable', 'loss_provenance':{'method':'deterministic evaluation reuse after verified final-parameter and prepared-input hashes',
            'origin_result':str(result_path), 'origin_result_sha256':hashlib.sha256(raw).hexdigest(),
            'trained_parameter_sha256':full['trained_parameter_sha256'],
            'validation_input_sha256':full['validation_input_sha256'],
            'implementation_sha256':full['implementation_sha256'],
            'evaluation_context':full['evaluation_context'],
            'origin_validation_seconds':full['validation_seconds'],
            'origin_provenance':full['provenance']},
            'per_user':full['per_user'], 'evaluation':full['evaluation']}


def origin_request(controller_root):
    """Origin case directory named by the launcher, or None. Only controller artifacts are accepted."""
    value = os.environ.get(ORIGIN_ENVIRONMENT)
    if not value or not controller_root:
        return None
    controller = Path(controller_root).resolve()
    origin = (controller / value).resolve()
    if not origin.is_relative_to(controller / 'artifacts') or not (origin / 'result.json').is_file():
        return {'status': 'refused', 'reasons': [f'Origin outside controller artifacts or without result: {value}']}
    return origin


def precheck(report, origin_directory):
    """Cheap equality screen before preparing held-out inputs; the full proof still runs afterwards."""
    full = json.loads((Path(origin_directory) / 'result.json').read_text(encoding='utf-8'))
    unequal = [field for field in REPORT_FIELDS if field not in ('validation_input_sha256', 'evaluation_context')
               and (report.get(field) in (None, '') or report.get(field) != full.get(field))]
    return [f'Unequal before input proof: {field}' for field in unequal]
