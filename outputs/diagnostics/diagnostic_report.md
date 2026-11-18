# PathSnap — Evaluator Diagnostic Report

## 1. Dataset

This report analyzes 10 evaluator GPS trips.
Candidate generation uses k=6 candidates per observation.

## 2. Overall Evaluation

| Metric | Raw ML | Viterbi |
|---|---:|---:|
| Weak-label agreement on covered observations | 55.24% | 70.66% |
| Connected-step fraction | 77.54% | 96.46% |
| Reference-arc recall (set-based) | 75.33% | 89.66% |

These metrics are unweighted averages across trips.

## 3. Per-Trip Diagnostics

| Trip | Observations | Coverage | Median candidate distance | Reference within 25 m |
|---|---:|---:|---:|---:|
| 00000015 | 6,854 | 4.92% | 5.26 m | 96.97% |
| 00000017 | 1,346 | 92.87% | 4.10 m | 100.00% |
| 00000023 | 458 | 80.57% | 6.40 m | 99.78% |
| 00000030 | 654 | 94.19% | 3.84 m | 99.24% |
| 00000049 | 2,642 | 97.05% | 4.86 m | 99.92% |
| 00000059 | 1,031 | 95.34% | 4.10 m | 100.00% |
| 00000067 | 916 | 50.44% | 4.05 m | 99.67% |
| 00000073 | 1,144 | 98.34% | 4.02 m | 99.83% |
| 00000087 | 1,399 | 95.93% | 4.36 m | 99.64% |
| 00000094 | 3,230 | 16.47% | 3.86 m | 99.88% |

## 4. Key Findings

- Lowest candidate coverage: trip `00000015` (4.92%).
- Highest candidate coverage: trip `00000073` (98.34%).

Low exact-arc candidate coverage does not necessarily imply large geometric distance from the reference route.

Possible explanations include heuristic alignment differences, arc identity ambiguity, or candidate-selection limitations. These causes have not been independently verified.

## 5. Limitations

- Weak-label agreement is not independently annotated GPS-point accuracy.
- Candidate coverage uses heuristic monotonic labels.
- Reference-arc recall ignores traversal order and additional predicted arcs.
- Connected-step fraction measures direct adjacency, not complete network-path feasibility.
- Exact-search diagnostics use sampled observations.

## 6. Generated Artifacts

- `comparison.csv`
- `comparison.png`
- `diagnostic_summary_all_trips.csv`
- Per-trip observation CSV files
- Per-trip summary JSON files
