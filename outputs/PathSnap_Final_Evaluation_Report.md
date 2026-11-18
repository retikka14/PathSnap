# PathSnap — Technical Evaluation Report (Generated from actual outputs)

## Objective
Reconstruct a plausible sequence of road segments from noisy GPS observations using supervised candidate scoring and sequential Viterbi decoding.

## Dataset and methodology
Training uses the supplied training trips; student-test trips are used for evaluation. Candidate segments are found using midpoint KD-tree preselection followed by point-to-segment distance ranking. A heuristic monotonic alignment to the provided chronological route creates **weak per-observation labels**. Five classifiers are compared using grouped cross-validation. Bayesian tuning uses training-trip validation. The trained classifier produces candidate emission scores; Viterbi penalizes non-connected transitions.

## Model comparison
| model | cv_macro_f1 | macro_f1 | top1_accuracy | log_loss |
| --- | --- | --- | --- | --- |
| logistic | 0.6962 | 0.6672 | 0.3730 | 0.4819 |
| decision_tree | 0.6402 | 0.6160 | 0.3717 | 0.2588 |
| random_forest | 0.6393 | 0.6251 | 0.3741 | 0.2395 |
| hist_gradient_boosting | 0.6388 | 0.6274 | 0.3748 | 0.2405 |
| extra_trees | 0.5877 | 0.5925 | 0.3731 | 0.2453 |

## Bayesian optimisation
Number of logged trials: **2**. See `outputs/bayesian_trials.csv` for all actual parameter trials.

## Route reconstruction results
Averages reported by the evaluation script over **10** student-test trips:

| metric | value |
| --- | --- |
| candidate_coverage_proxy | 0.8249 |
| raw_weak_label_accuracy_on_covered | 0.5742 |
| viterbi_weak_label_accuracy_on_covered | 0.6874 |
| raw_connected_step_fraction | 0.8023 |
| viterbi_connected_step_fraction | 0.9792 |
| raw_reference_arc_recall_set | 0.7891 |
| viterbi_reference_arc_recall_set | 0.9097 |
## Per-trip results
| trip_id | observations | candidate_coverage_proxy | raw_connected_step_fraction | viterbi_connected_step_fraction |
| --- | --- | --- | --- | --- |
| 0.0000 | 2503.0000 | 0.9608 | 0.9017 | 0.9892 |
| 7.0000 | 1354.0000 | 0.9978 | 0.9231 | 1.0000 |
| 9.0000 | 1049.0000 | 0.9247 | 0.6345 | 0.9666 |
| 40.0000 | 2995.0000 | 0.9559 | 0.8948 | 0.9957 |
| 52.0000 | 262.0000 | 0.9771 | 0.7739 | 0.9693 |
| 56.0000 | 3958.0000 | 0.0432 | 0.7630 | 0.9687 |
| 65.0000 | 768.0000 | 0.7318 | 0.7953 | 0.9765 |
| 76.0000 | 1486.0000 | 0.9482 | 0.7333 | 0.9616 |
| 77.0000 | 3911.0000 | 0.7241 | 0.8573 | 0.9923 |
| 91.0000 | 874.0000 | 0.9851 | 0.7457 | 0.9725 |

## Diagnostic comparison
| trip | candidate_coverage_proxy | median_nearest_reference_m | reference_within_50m_fraction | sampled_midpoint_search_miss_fraction |
| --- | --- | --- | --- | --- |
| 56.0000 | 0.0432 | 4.4979 | 0.9932 | 0.0417 |
| 7.0000 | 0.9978 | 4.5084 | 1.0000 | 0.0000 |

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
