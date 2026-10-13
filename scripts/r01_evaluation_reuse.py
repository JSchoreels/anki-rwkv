"""F67 fail-closed evaluation reuse proof; no runner integration or protocol changes."""
import hashlib
import json
from pathlib import Path

REPORT_FIELDS = ('model', 'torch', 'cuda', 'gpu', 'master_dtype', 'float32_matmul_precision',
                 'cuda_matmul_allow_tf32', 'implementation_sha256', 'trained_parameter_sha256',
                 'validation_input_sha256', 'evaluation_users', 'config', 'evaluation_context')
PROTOCOL_FIELDS = ('dataset_revision', 'cohort_sha256', 'evaluation_rows', 'evaluation', 'holdout', 'seeds')
CONTEXT_FIELDS = ('driver_version', 'nvrtc_version', 'backend', 'deterministic_algorithms',
                  'cublas_workspace_config', 'evaluation_dtype', 'prepared_input_sha256', 'evaluation_source_sha256')

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
    if full.get('status') != 'complete' or full.get('smoke_only') is not False:
        failures.append('Full evaluation origin must be a completed non-smoke case')
    for field in REPORT_FIELDS:
        if field not in candidate_report or field not in full or candidate_report[field] in (None, '') or candidate_report[field] != full[field]:
            failures.append(f'Missing or unequal report proof: {field}')
    for report in (candidate_report, full):
        context = report.get('evaluation_context')
        if not isinstance(context, dict) or any(field not in context or context[field] in (None, '') for field in CONTEXT_FIELDS):
            failures.append('Missing complete evaluation runtime/source manifest')
        elif context['prepared_input_sha256'] != report.get('validation_input_sha256') or not context['deterministic_algorithms']:
            failures.append('Evaluation manifest lacks deterministic exact input proof')
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
