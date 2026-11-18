"""PathSnap student-test evaluation; reference routes are used ONLY here, never in API inference."""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np
from joblib import load
from src.core import trip_files, candidate_features, derived_labels, viterbi

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'Student folder' / 'student_test'
OUT = ROOT / 'outputs'

def connected_fraction(path, arcs):
    if len(path) < 2:
        return float('nan')
    a, b = arcs[path[:-1]], arcs[path[1:]]
    return float(np.mean((path[:-1] == path[1:]) | (a[:,1] == b[:,0])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='0 = all student-test trips')
    args = parser.parse_args()
    artifact = load(OUT / 'model.joblib')
    model, k = artifact['model'], int(artifact['k'])
    folders = sorted(p for p in DATA.iterdir() if p.is_dir())
    if args.limit: folders = folders[:args.limit]
    rows = []
    for folder in folders:
        track, nodes, arcs, route = trip_files(folder, require_route=True)
        cand, feats, _, xy = candidate_features(track, nodes, arcs, k=k)
        scores = model.predict_proba(feats.reshape(-1, feats.shape[-1]))[:,1].reshape(len(track), k)
        scores = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
        raw = cand[np.arange(len(track)), np.argmax(scores, axis=1)]
        decoded = viterbi(scores, cand, arcs, xy, track[:,2])
        # Proxy labels only; generated using reference route and not independent GPS-point truth.
        weak_y, hits = derived_labels(track, nodes, arcs, route, cand)
        # Reconstruct proxy target from the one-hot row when it is among candidates.
        usable = hits.astype(bool)
        weak_target = cand[np.arange(len(track)), np.argmax(weak_y, axis=1)]
        raw_proxy = float(np.mean(raw[usable] == weak_target[usable])) if usable.any() else float('nan')
        decoded_proxy = float(np.mean(decoded[usable] == weak_target[usable])) if usable.any() else float('nan')
        ref_unique = set(map(int, route))
        raw_unique = set(map(int, raw))
        decoded_unique = set(map(int, decoded))
        row = {
            'trip_id': folder.name, 'observations':len(track),
            'candidate_coverage_proxy':float(np.mean(hits)),
            'raw_weak_label_accuracy_on_covered':raw_proxy,
            'viterbi_weak_label_accuracy_on_covered':decoded_proxy,
            'raw_connected_step_fraction':connected_fraction(raw, arcs),
            'viterbi_connected_step_fraction':connected_fraction(decoded, arcs),
            'raw_reference_arc_recall_set':len(raw_unique & ref_unique)/len(ref_unique),
            'viterbi_reference_arc_recall_set':len(decoded_unique & ref_unique)/len(ref_unique),
            'raw_unique_arcs':len(raw_unique), 'viterbi_unique_arcs':len(decoded_unique),
            'reference_unique_arcs':len(ref_unique),
        }
        rows.append(row)
        print(f"{folder.name}: GPS={len(track)} coverage={row['candidate_coverage_proxy']:.3f} "
              f"connected raw={row['raw_connected_step_fraction']:.3f} "
              f"viterbi={row['viterbi_connected_step_fraction']:.3f}", flush=True)
    if not rows: sys.exit('No student-test trips found')
    OUT.mkdir(exist_ok=True)
    with (OUT/'route_evaluation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={k:float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k not in ('trip_id','observations')}
    summary['trip_count']=len(rows)
    summary['limitations']=['Weak-label accuracy is a proxy, not independently annotated per-point accuracy.',
                            'Reference-arc recall ignores segment order and extra predicted arcs.',
                            'Connected-step fraction tests direct adjacency, not full road-network path feasibility.']
    (OUT/'route_evaluation_summary.json').write_text(json.dumps(summary,indent=2))
    print('\nSUMMARY',json.dumps(summary,indent=2))
    print('SAVED outputs/route_evaluation.csv and outputs/route_evaluation_summary.json')
if __name__=='__main__':main()
