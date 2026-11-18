# PathSnap

Reconstructs the road segments behind a noisy GPS track: candidate arcs are scored
by a trained classifier, then a connectivity-aware Viterbi pass picks one arc per
observation.

## What is in here

| Path | Purpose |
| --- | --- |
| `src/core.py` | Trip loading, candidate generation, weak labels, Viterbi decoder |
| `src/train.py` | Trains and compares models, writes `outputs/model.joblib` |
| `src/api.py` | FastAPI service: `/health`, `/trips`, `/predict/{trip_id}` |
| `app.py` | One-file viewer that proxies the API and draws the route |
| `outputs/` | Trained model, evaluation CSVs and reports |
| `evaluate_routes.py`, `diagnose_paths.py`, `build_final_report.py` | Offline evaluation |

The dataset (`data/Student folder/`) is not in the repository; place it there before
running anything.

## Run the backend

```bash
python -m pip install fastapi "uvicorn[standard]" numpy scipy scikit-learn joblib
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Check it: <http://localhost:8000/health> → `{"status":"ok","model_ready":true,...}`.

## Run the viewer

```bash
python app.py --api http://localhost:8000
```

A browser opens at <http://127.0.0.1:5500>. Pick a trip and press **Reconstruct
route** to see the GPS track (blue), the decoded segments (green) and start/end
markers. `--port` and `--host` change where the viewer listens.

### Two laptops

Start the backend on the machine that holds the data, with `--host 0.0.0.0`, then
point the viewer at its LAN address:

```bash
python app.py --api http://192.168.1.42:8000
```

Both machines must be on the same network and the backend machine must allow TCP
8000 through its firewall. Test `http://192.168.1.42:8000/health` in a browser
first; the viewer reports the exact connection error if it cannot reach the API.

## API

```
GET /health                {"status": "ok", "model_ready": true}
GET /trips                 {"partition": "student_test", "trips": ["00000007", ...]}
GET /predict/{trip_id}     {"trip_id", "observations", "gps": [[lat, lon], ...],
                            "raw_arc_ids", "decoded_arc_ids", "predicted_segments"}
```

`gps` and the segment endpoints are `[latitude, longitude]`. `raw_arc_ids` and
`decoded_arc_ids` hold one arc id per observation; `predicted_segments` holds one
segment per observation and may repeat. Reference `.route` files are never read by
the API — they are only used by the offline evaluation scripts.

## Notes

- Models are trained on weakly derived labels (monotonic alignment to the reference
  route), so the metrics in `outputs/` are proxy estimates, not verified per-point
  accuracy.
- `PATHSNAP_EVAL_SPLIT` selects the partition served by `/trips` and `/predict`
  (default `student_test`).
