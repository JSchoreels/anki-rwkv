"""Augment standard closure handoff with inherited campaign and alternative schedules."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[3]
base=root/'artifacts/retraining/gpu-retrain-r01'
d=root/'artifacts/diagnostics/r01-round-0063-0065'
previous=json.loads((base/'codex-round-0060-0062-handoff.json').read_text())
p=base/'codex-round-0063-0065-handoff.json'
h=json.loads(p.read_text())
audit=json.loads((d/'round-audit.json').read_text())
stack=previous['stack_candidates']
stack[1]['caveat']='0061 mean dominated by one +15.964% fast-state block with faster preparation/validation. Combined original F58 + F58e checked in 0063 but rejected at two blocks; neither standalone component is independently promoted.'
rejected=list(previous['rejected_candidates'])
rejected_modes=list(previous['rejected_option_overrides']['recurrence_load_pipeline'])
for a in audit:
 n=a['iteration']; c=json.loads((base/f'iteration-{n}/comparison.json').read_text())['comparisons']['expertium']
 if c['stack_candidate']:
  mode={'0063':5,'0064':7,'0065':8}[n]
  stack.append({'iteration':n,'track':c['track'],'option_overrides':{'recurrence_load_pipeline':mode,'recurrence_prefetch':n=='0063'},'mean_gain_percent':a['mean_percent'],'lower_bound_percent':a['lower_bound_percent'],'decision':'positive unproven at six-block cap','caveat':'Modes 5/7/8 are alternative forward schedules; no additive effect or compatible combination is established.'})
 elif not c['accepted_main']:
  rejected.append(n+' '+c['track'])
  if n!='0063': rejected_modes.append({'0064':7,'0065':8}[n])
h.update(next_iteration='0066',previous_argument='0065',loss_ceiling_fraction=previous['loss_ceiling_fraction'],stack_candidates=stack,rejected_candidates=rejected,rejected_option_overrides={'recurrence_load_pipeline':rejected_modes},rejected_bundles=[{'iteration':'0063','option_overrides':{'recurrence_prefetch':True,'recurrence_load_pipeline':5},'decision':'reject; do not infer rejection of either standalone component'}],candidate_preparation='artifacts/diagnostics/r01-round-0063-0065/preparation.md',batch_decisions=audit,worker_stop='Completed exactly 0063-0065; parent controls chats and shutdown',pending_components=['F62 whole-group AdamW preserving None gradients/per-parameter steps/signed zeros','F34 exact eight-way shift/lerp preserving dtype rounding and backward reduction order','F49 genuinely independent deferred overlap only after memory-neutral preparation','Substantive F51/F52 preserving dropout and split weight-gradient order'],p3b_attribution=previous['p3b_attribution'],schedule_compatibility='Modes 5, 7 and 8 are mutually exclusive forward schedules. Original F58 is a distinct backward scalar prefetch. Do not count a replay of rejected 0063 or assume retained gains add.',report_repairs='0064/0065 Windows Errno 22 opening run log; explicit report-only resumes preserved completed training/profile and failed attempts.')
assert h['round_iteration_budget']==23 and h['completed_candidate_iterations']==18 and h['remaining_authorized_iterations']==5
assert h['accepted_options']==previous['accepted_options']
p.write_text(json.dumps(h,indent=2)+'\n')
print('handoff: accepted 0042; 18/23 complete; next 0066/0065; ceiling 0.001')
