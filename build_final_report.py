from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
def fmt(v):
    try:return f'{float(v):.4f}'
    except (TypeError,ValueError):return str(v)
def table(df,cols=None):
    if cols is not None:df=df[cols]
    heads=list(df.columns)
    lines=['| '+' | '.join(heads)+' |','| '+' | '.join(['---']*len(heads))+' |']
    for _,r in df.iterrows():lines.append('| '+' | '.join(fmt(r[c]) for c in heads)+' |')
    return '\n'.join(lines)
def main():
    required=['model_comparison.csv','bayesian_trials.csv','route_evaluation.csv','route_evaluation_summary.json']
    missing=[n for n in required if not (OUT/n).exists()]
    if missing:raise SystemExit('Missing outputs: '+', '.join(missing))
    models=pd.read_csv(OUT/'model_comparison.csv')
    trials=pd.read_csv(OUT/'bayesian_trials.csv')
    routes=pd.read_csv(OUT/'route_evaluation.csv')
    summary=json.loads((OUT/'route_evaluation_summary.json').read_text())
    diagnostics=OUT/'diagnostics'/'comparison.csv'
    diagnostic_section=table(pd.read_csv(diagnostics),['trip','candidate_coverage_proxy','median_nearest_reference_m','reference_within_50m_fraction','sampled_midpoint_search_miss_fraction']) if diagnostics.exists() else 'Diagnostics not run yet.'
    keys=['candidate_coverage_proxy','raw_weak_label_accuracy_on_covered','viterbi_weak_label_accuracy_on_covered','raw_connected_step_fraction','viterbi_connected_step_fraction','raw_reference_arc_recall_set','viterbi_reference_arc_recall_set']
    report=f'''# PathSnap — Technical Evaluation Report (Generated from actual outputs)

## Objective
Reconstruct a plausible sequence of road segments from noisy GPS observations using supervised candidate scoring and sequential Viterbi decoding.

## Dataset and methodology
Training uses the supplied training trips; student-test trips are used for evaluation. Candidate segments are found using midpoint KD-tree preselection followed by point-to-segment distance ranking. A heuristic monotonic alignment to the provided chronological route creates **weak per-observation labels**. Five classifiers are compared using grouped cross-validation. Bayesian tuning uses training-trip validation. The trained classifier produces candidate emission scores; Viterbi penalizes non-connected transitions.

## Model comparison
{table(models[['model','cv_macro_f1','macro_f1','top1_accuracy','log_loss']])}

## Bayesian optimisation
Number of logged trials: **{len(trials)}**. See `outputs/bayesian_trials.csv` for all actual parameter trials.

## Route reconstruction results
Averages reported by the evaluation script over **{summary['trip_count']}** student-test trips:

{table(pd.DataFrame([{'metric': key, 'value': summary[key]} for key in keys]))}
## Per-trip results
{table(routes[['trip_id','observations','candidate_coverage_proxy','raw_connected_step_fraction','viterbi_connected_step_fraction']])}

## Diagnostic comparison
{diagnostic_section}

## Interpretation
The connected-step fraction checks whether consecutive predictions are the same arc or directly linked. It does not prove that the entire path is drivable. The weak-label accuracy is conditional on the inferred target being among the candidates and is not independently verified GPS-point accuracy. Set-based reference arc recall ignores ordering, repeated arcs and false positives. Results should not be represented as ground-truth end-to-end route accuracy.

## Limitations and next experiments
1. Check candidate recall using exact point-to-segment distances on representative samples; midpoint preselection can miss long segments.
2. Compare heuristic alignment against geometry proximity and investigate low-coverage trips, particularly 00000056.
3. Implement road-network shortest-path feasibility and ordered-route evaluation.
4. Retest on held-out trips without tuning on evaluator-only data.
5. Record the actual two-laptop deployment test and demo screenshots separately.

## Reproducibility
Inputs: `outputs/model_comparison.csv`, `outputs/bayesian_trials.csv`, `outputs/route_evaluation.csv`, `outputs/route_evaluation_summary.json` and optional `outputs/diagnostics/comparison.csv`.
'''
    target=OUT/'PathSnap_Final_Evaluation_Report.md';target.write_text(report)
    print('SAVED',target)
if __name__=='__main__':main()
