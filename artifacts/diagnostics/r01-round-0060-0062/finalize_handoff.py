"""Augment the standard closure handoff with inherited explicit stack state."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[3]
base = Path('artifacts/retraining/gpu-retrain-r01')
previous = json.loads((root/base/'codex-round-0057-0059-handoff.json').read_text(encoding='utf-8'))
path = root/base/'codex-round-0060-0062-handoff.json'
handoff = json.loads(path.read_text(encoding='utf-8'))
audit = json.loads((root/'artifacts/diagnostics/r01-round-0060-0062/round-audit.json').read_text(encoding='utf-8'))
handoff.update(next_iteration='0063', previous_argument='0062', loss_ceiling_fraction=previous['loss_ceiling_fraction'],
    stack_candidates=[*previous['stack_candidates'], {'iteration':'0061', 'track':'E33-F48-F50-F58e', 'option_overrides':{'recurrence_load_pipeline':5}, 'mean_gain_percent':audit[1]['mean_percent'], 'lower_bound_percent':audit[1]['lower_bound_percent'], 'decision':'positive unproven at six-block cap', 'caveat':'One +15.964% block dominates; candidate preparation and validation faster too. No combined F58+F58e check yet.'}],
    rejected_candidates=[*previous['rejected_candidates'], '0060 E33-F48-F50-F58d', '0062 E33-F48-F50-F58f'],
    rejected_option_overrides={'recurrence_load_pipeline':[1,2,3,4,6]}, candidate_preparation='artifacts/diagnostics/r01-round-0060-0062/preparation.md',
    batch_decisions=audit, worker_stop='Completed exactly 0060-0062; parent controls next chat and shutdown',
    pending_components=['F62 whole-group AdamW preserving None gradients/steps/signed zeros', 'F34 exact eight-way shift/lerp forward and original backward reductions', 'Combined original F58 + new F58e, only after compatible combined checks', 'F49 independent deferred overlap if memory-neutral', 'Substantive F51/F52 preserving dropout and split weight-gradient order'],
    p3b_attribution={'status':'exact diagnostic, inconclusive', 'result':'artifacts/diagnostics/r01-round-0060-0062/twin/result.json', 'pairs':20, 'mean_log_gain_percent':-0.22556948161998266, 'promotion_rule_unchanged':True})
assert handoff['round_iteration_budget'] == 23 and handoff['completed_candidate_iterations'] == 15 and handoff['remaining_authorized_iterations'] == 8
assert handoff['accepted_options'] == previous['accepted_options']
path.write_text(json.dumps(handoff, indent=2)+'\n', encoding='utf-8')
print('handoff: accepted 0042; 15/23 complete; next 0063/0062; retained 0048 and 0061; ceiling 0.001')
