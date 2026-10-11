"""Report the five authorized R01 screens 0002–0006 and accepted references."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gpu_r01_policy import accepted_reference, read_reference
from scripts.report_gpu_r01_next import describe

BASE = ROOT / 'artifacts/retraining/gpu-retrain-r01'


def main():
    iterations = ['0002', '0003', '0004', '0005', '0006']
    reports = {iteration: json.loads((BASE / f'iteration-{iteration}/comparison.json').read_text(encoding='utf-8'))
               for iteration in iterations}
    if not all(report['status'].startswith('complete') for report in reports.values()):
        raise ValueError('Five completed screens are required')
    accepted = {}
    for model, iteration in (('baseline', '0005'), ('expertium', '0006')):
        metadata = accepted_reference(BASE / f'iteration-{iteration}', model, ROOT)
        case, path = read_reference(metadata, ROOT)
        original_path = BASE / f'iteration-0002/{model}/control/result.json'
        original = json.loads(original_path.read_text(encoding='utf-8'))
        keys = ('evaluation', 'trained_parameter_sha256', 'initial_parameter_sha256',
                'validation_input_sha256', 'coverage', 'updates', 'review_presentations',
                'training_curve_labels', 'training_instant_labels')
        if any(case[key] != original[key] for key in keys):
            raise ValueError('Series retained-result parity claim is not supported')
        accepted[model] = {'reference': metadata, 'track': case['track'], 'options': case['options'],
                           'training_seconds': case['training_seconds'], 'reviews_per_second': case['reviews_per_second'],
                           'gain_vs_original_percent': 100 * (case['reviews_per_second'] / original['reviews_per_second'] - 1),
                           'evaluation': case['evaluation'], 'updates': case['updates'],
                           'review_presentations': case['review_presentations'],
                           'final_parameter_sha256': case['trained_parameter_sha256'],
                           'experiment_provenance': case['provenance']}
    result = {'status': 'five completed preliminary screens; both models stopped',
              'iterations': iterations, 'accepted': accepted,
              'stop_streaks': {'baseline': reports['0005']['comparisons']['baseline']['no_20_percent_gain_streak'],
                              'expertium': reports['0006']['comparisons']['expertium']['no_20_percent_gain_streak']},
              'evaluation_users': 20, 'training_users': 80, 'broader_validation': 'pending; not authorized',
              'pending_ideas': ['F17 (larger training batches)'],
              'timing_repeats': 'one timing per configuration; small incremental gains are not established as repeatable'}
    if result['stop_streaks'] != {'baseline': 3, 'expertium': 3}:
        raise ValueError('Both recorded stopping thresholds must be reached')
    (BASE / 'five-iteration-summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    lines = ['# Five completed GPU retraining screens', '',
             'Completed iterations **0002–0006**, rereading the loop between rounds. Historical partial iteration 0001 is preserved and excluded from this count. Alex stopped before iteration 0006; Expertium completed the fifth screen. Both models now have three consecutive gains below the 20% target.', '',
             'Every full case reused the fixed **80 training users**, **20 unseen held-out users**, **7,654,975 training reviews**, **868,370 evaluation reviews**, one epoch and seed 2025. All accepted variants retain exactly the original final weights, held-out inputs, review/label coverage and all four binary loss scores. The final accepted Alex case has 507 updates; Expertium has 276. These are preliminary screens; broader validation remains pending.', '',
             '## Retained configurations', '',
             '| Model | Retained improvements | Training seconds | Reviews/s | Speed Δ vs original | Curve weighted | Curve user | Instant weighted | Instant user |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for model, item in accepted.items():
        e = item['evaluation']
        lines.append(f"| {'Alex (default)' if model == 'baseline' else 'Expertium (light)'} | {describe(item['track'])} | {item['training_seconds']:.3f} | {item['reviews_per_second']:,.0f} | {item['gain_vs_original_percent']:+.2f}% | {e['curve_log_loss_label_weighted']:.9f} | {e['curve_log_loss_user_average']:.9f} | {e['instant_log_loss_label_weighted']:.9f} | {e['instant_log_loss_user_average']:.9f} |")
    lines += ['', 'The training timer includes user loads, preparation, worker startup/shutdown, transfers, updates and logs. Setup, discarded primers, saves and evaluation are separate in each raw result. Small incremental gains from GPU encodings, packed communication and overlapping transfers have one timing each; they do not establish timing repeatability.', '',
              '## Iteration decisions', '', '| Iteration | Alex candidate | Alex speed Δ | Expertium candidate | Expertium speed Δ | Decision |', '|---|---|---:|---|---:|---|']
    for iteration, report in reports.items():
        cells = []
        for model in ('baseline', 'expertium'):
            comparison = report['comparisons'].get(model)
            if comparison is None:
                cells += ['Stopped by the three-miss rule', '—']
            else:
                gain = comparison.get('throughput_gain_percent')
                if gain is None:
                    gain = comparison['references']['previous']['throughput_gain_percent']
                cells += [describe(comparison['track']), f'{gain:+.2f}%']
        decisions = {'0002': 'Retain ordered user prefetch for both models; exact results.',
                     '0003': 'Reject Alex reduced synchronization for speed; retain Expertium eight-worker preparation; exact results.',
                     '0004': 'Reject Alex fused shifts for speed; retain Expertium GPU encodings; exact results.',
                     '0005': 'Reject Alex reduced padding: slower and Instant losses exceed 1%; stop Alex. Retain Expertium packed communication; exact results.',
                     '0006': 'Retain Expertium transfer overlap; exact results. Stop Expertium after its third target miss.'}
        lines.append(f"| [{iteration}](iteration-{iteration}/summary.md) | {' | '.join(cells)} | {decisions[iteration]} |")
    lines += ['', 'Speed deltas use the previous accepted run; iteration 0002 uses its original no-prefetch control. Iteration 0003 also measured fresh controls: Expertium gained 19.82% against its fresh control and 26.63% against its prior accepted run. The revised comparator and stopping-rule transition are preserved in [the iteration 0004 amendment](iteration-0004/reference-reuse-amendment.json).', '',
              'Reduced padding cut Alex padded positions by 17.64% but lost 13.06% throughput. Weighted Instant loss rose 1.48% and user-average Instant loss rose 1.73%; weighted Curve improved 3.77% and user-average Curve improved 4.84%. The four separate gates reject this tradeoff. No faster candidate regressed quality, so no new fastmode branch was needed.', '',
              '## Evidence and commits', '',
              '[Per-run log](../../../docs/GPU_OPTIMIZATIONS_LOG.md), [optimization ledger](../../../docs/GPU_OPTIMIZATIONS.md), [structured series result](five-iteration-summary.json). Raw results, source archives, resource profiles and checkpoints are linked from each iteration report. Failed source-change, sandbox-primer and telemetry attempts remain visible and have no final throughput or quality score.', '']
    for model, item in accepted.items():
        relative = Path(item['reference']['path']).relative_to(BASE.relative_to(ROOT)).as_posix()
        lines.append(f"- {'Alex' if model == 'baseline' else 'Expertium'} retained result: [{describe(item['track'])}]({relative}); original process commit `{item['experiment_provenance']['git_commit']}`, dirty `{item['experiment_provenance']['git_dirty']}` with its preserved source archive.")
    lines += ['', 'Accepted implementations and post-experiment commit records are in the individual reports. Frozen isolated source execution preserved the training pipeline despite concurrent checkout edits; the isolated Git identity and original source lineage are recorded separately. The pending F17 (larger training batches) idea remains unmeasured and was not run.', '']
    (BASE / 'five-iteration-summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({model: {'track': item['track'], 'reviews_per_second': item['reviews_per_second'],
                              'gain_vs_original_percent': item['gain_vs_original_percent']}
                      for model, item in accepted.items()}))


if __name__ == '__main__':
    main()
