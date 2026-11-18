#!/usr/bin/env python3
"""PathSnap one-file viewer.

Serves a single-page UI and proxies the PathSnap FastAPI backend that runs on the
other laptop, so the browser only ever talks to this process (no CORS, no mixed
origins, and the friend's LAN IP lives in one place).

Run it:

    python app.py --api http://192.168.1.42:8000

    # or edit BACKEND_URL below and just:
    python app.py

Then open http://127.0.0.1:5500 (it opens automatically unless --no-browser).
Python 3.9+, standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

# --------------------------------------------------------------------------
# Set your friend's laptop address here (override with --api or PATHSNAP_API).
# --------------------------------------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000"

TRIP_ID = re.compile(r"\d{1,12}")
TIMEOUTS = {"health": 10, "trips": 20, "predict": 180}


# --------------------------------------------------------------------------
# HTTP proxy + static page
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    backend = BACKEND_URL
    server_version = "PathSnap/1.0"

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        route = urlsplit(self.path).path
        if route in ("/", "/index.html"):
            self._send_html(PAGE)
            return
        if route == "/api/health":
            self._proxy("/health", "health")
            return
        if route == "/api/trips":
            self._proxy("/trips", "trips")
            return
        if route.startswith("/api/predict/"):
            trip_id = unquote(route[len("/api/predict/") :])
            if not TRIP_ID.fullmatch(trip_id):
                self._send_json(400, {"detail": "trip_id must be 1-12 digits"})
                return
            self._proxy(f"/predict/{trip_id}", "predict")
            return
        self._send_json(404, {"detail": "Not found"})

    # -- helpers ----------------------------------------------------------

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-PathSnap-Backend", self.backend)
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, path: str, kind: str) -> None:
        url = self.backend.rstrip("/") + path
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUTS[kind]) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as error:
            # Forward the API's own error body (e.g. 404 unknown trip, 503 no model).
            status = error.code
            raw = error.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            self._send_json(
                502,
                {
                    "detail": (
                        f"Cannot reach the PathSnap API at {self.backend}: {reason}. "
                        "Check the IP, that both laptops are on the same network, and "
                        "that the backend was started with --host 0.0.0.0."
                    )
                },
            )
            return

        try:
            payload = json.loads(raw.decode("utf-8"))
            body = json.dumps(payload).encode("utf-8")
        except (ValueError, UnicodeDecodeError):
            body = raw
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-PathSnap-Backend", self.backend)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[viewer] {self.address_string()} {fmt % args}\n")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathSnap</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#070b14; color:#e6eefc;
         font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { display:flex; align-items:center; justify-content:space-between; gap:16px;
           flex-wrap:wrap; padding:14px 18px; border-bottom:1px solid #1e2d4a; background:#0d1526; }
  h1 { font-size:17px; margin:0; letter-spacing:.2px; }
  header p { margin:0; font-size:12px; color:#8fa3c4; }
  .pill { display:inline-flex; align-items:center; gap:8px; font-size:12.5px; color:#8fa3c4;
          border:1px solid #1e2d4a; background:#111c31; border-radius:999px; padding:6px 12px; }
  .pill i { width:8px; height:8px; border-radius:50%; background:#64789b; display:inline-block; }
  .pill.ok { color:#bbf7d0; border-color:#166534; }
  .pill.ok i { background:#22c55e; }
  .pill.err { color:#fecaca; border-color:#7f1d1d; }
  .pill.err i { background:#ef4444; }
  main { display:grid; grid-template-columns:290px minmax(0,1fr); gap:16px; padding:16px; }
  .card { background:#0d1526; border:1px solid #16233a; border-radius:14px; }
  aside { padding:14px; display:grid; gap:14px; align-content:start; }
  label { display:block; font-size:12.5px; color:#8fa3c4; margin-bottom:6px; }
  select, button { font:inherit; font-size:13.5px; color:#e6eefc; background:#111c31;
                   border:1px solid #1e2d4a; border-radius:9px; padding:9px 11px; }
  select { width:100%; }
  button { cursor:pointer; }
  button:hover:not(:disabled) { background:#16233c; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.primary { width:100%; margin-top:10px; background:#2563eb; border-color:#2563eb;
                   color:#f8fbff; font-weight:600; }
  .checks { display:grid; gap:8px; }
  .checks label { display:flex; align-items:center; gap:9px; margin:0; color:#e6eefc;
                  font-size:13.5px; cursor:pointer; }
  .sw { width:18px; height:3px; border-radius:2px; display:inline-block; }
  .sw.gps { background:#3b82f6; } .sw.pred { background:#22c55e; }
  dl { margin:0; display:grid; gap:1px; background:#16233a; border-radius:9px; overflow:hidden; }
  dl div { display:flex; justify-content:space-between; gap:10px; padding:7px 10px; background:#0d1526; }
  dt { color:#8fa3c4; font-size:12.5px; } dd { margin:0; font-family:ui-monospace, monospace;
       font-size:13px; text-align:right; font-variant-numeric:tabular-nums; }
  .hint { font-size:11.5px; color:#64789b; margin:0; }
  .mapcard { padding:0; overflow:hidden; display:flex; flex-direction:column; min-height:540px; }
  .maphead { display:flex; align-items:center; justify-content:space-between; gap:12px;
             padding:11px 14px; border-bottom:1px solid #16233a; flex-wrap:wrap; }
  .maphead strong { font-weight:600; font-size:14px; }
  .tools { display:flex; gap:7px; }
  .tools button { padding:6px 10px; font-size:12.5px; }
  #wrap { position:relative; flex:1; min-height:430px; background:#060a12; }
  canvas { display:block; width:100%; height:100%; }
  .empty { position:absolute; inset:0; display:grid; place-content:center; text-align:center;
           color:#8fa3c4; font-size:13.5px; padding:20px; }
  .empty strong { display:block; color:#e6eefc; font-size:15px; margin-bottom:4px; }
  .loading { position:absolute; inset:0; display:grid; place-content:center; justify-items:center;
             gap:10px; background:rgba(6,10,18,.75); font-size:13.5px; color:#93c5fd; }
  .spin { width:32px; height:32px; border-radius:50%; border:2.5px solid #1e3a5f;
          border-top-color:#60a5fa; animation:spin .85s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .bar { display:flex; align-items:center; justify-content:space-between; gap:12px;
         padding:9px 14px; border-top:1px solid #16233a; font-size:12.5px; color:#8fa3c4; }
  .bar button { padding:3px 8px; font-size:12px; border:0; background:transparent; color:#8fa3c4;
                text-decoration:underline; }
  .err { margin:0 14px 12px; padding:10px 12px; border-radius:9px; border:1px solid #7f1d1d;
         background:#2a1214; color:#fecaca; font-size:12.5px; font-family:ui-monospace, monospace;
         white-space:pre-wrap; overflow-wrap:anywhere; }
  .err[hidden], .loading[hidden], .empty[hidden] { display:none; }
  footer { padding:0 18px 18px; font-size:11.5px; color:#64789b; }
  @media (max-width: 860px) { main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div>
    <h1>PathSnap</h1>
    <p>GPS observations &rarr; decoded road segments</p>
  </div>
  <div class="pill" id="pill"><i></i><span id="pillText">Connecting…</span></div>
</header>

<main>
  <aside class="card">
    <div>
      <label for="trip">Trip</label>
      <select id="trip"><option value="">Connecting…</option></select>
      <button class="primary" id="run" disabled>Reconstruct route</button>
      <p class="hint" id="backend" style="margin-top:8px"></p>
    </div>

    <div class="checks">
      <label><input type="checkbox" id="showGps" checked><span class="sw gps"></span> Raw GPS track</label>
      <label><input type="checkbox" id="showPred" checked><span class="sw pred"></span> Predicted segments</label>
    </div>

    <dl>
      <div><dt>Trip</dt><dd id="mTrip">—</dd></div>
      <div><dt>Observations</dt><dd id="mObs">—</dd></div>
      <div><dt>Decoded observations</dt><dd id="mDecoded">—</dd></div>
      <div><dt>Distinct decoded arcs</dt><dd id="mArcs">—</dd></div>
      <div><dt>Server inference</dt><dd id="mTime">—</dd></div>
    </dl>
  </aside>

  <section class="card mapcard">
    <div class="maphead">
      <strong>Route view</strong>
      <div class="tools">
        <button id="fit" disabled>Fit route</button>
        <button id="clear">Clear</button>
      </div>
    </div>
    <div id="wrap">
      <canvas id="map"></canvas>
      <div class="empty" id="empty">
        <div><strong>No route drawn yet</strong>Pick a trip, then run Reconstruct route.</div>
      </div>
      <div class="loading" id="loading" hidden><div class="spin"></div><span id="loadingText">Running Viterbi decode…</span></div>
    </div>
    <div class="bar"><span id="status">Connecting to the backend…</span></div>
    <div class="err" id="err" hidden></div>
  </section>
</main>

<footer>Predictions come from the server's classifier + Viterbi decoder. No basemap tiles, API keys, or accuracy claims.</footer>

<script>
const $ = (id) => document.getElementById(id);
const canvas = $('map');
const ctx = canvas.getContext('2d');
let data = null;
let view = null;

function status(text, kind) {
  $('status').textContent = text;
  $('pillText').textContent = kind === 'ok' ? 'Connected' : kind === 'err' ? 'Disconnected' : 'Connecting…';
  $('pill').className = 'pill' + (kind ? ' ' + kind : '');
}
function fail(message) {
  $('err').hidden = false;
  $('err').textContent = message;
  status(message, 'err');
}
function clearError() { $('err').hidden = true; $('err').textContent = ''; }

async function getJson(path) {
  const response = await fetch(path, { headers: { Accept: 'application/json' } });
  const text = await response.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch (e) { body = null; }
  if (!response.ok) {
    throw new Error((body && body.detail) || ('HTTP ' + response.status + ' from ' + path));
  }
  if (!body) throw new Error('Expected JSON from ' + path);
  return body;
}

async function connect() {
  clearError();
  status('Connecting to the backend…', '');
  try {
    const health = await getJson('api/health');
    const trips = await getJson('api/trips');
    $('backend').textContent = 'API: ' + (trips.partition || health.partition || 'unknown partition');
    const select = $('trip');
    select.innerHTML = '';
    const ids = Array.isArray(trips.trips) ? trips.trips : [];
    if (!ids.length) {
      select.innerHTML = '<option value="">No trips available</option>';
      fail('The backend listed no trips for partition "' + (trips.partition || '?') + '".');
      return;
    }
    for (const id of ids) {
      const option = document.createElement('option');
      option.value = String(id);
      option.textContent = String(id);
      select.append(option);
    }
    select.disabled = false;
    $('run').disabled = !health.model_ready;
    if (health.model_ready) {
      status(ids.length + ' trips ready · ' + (trips.partition || ''), 'ok');
    } else {
      fail('The backend is up but model.joblib is missing (model_ready: false).');
    }
  } catch (error) {
    $('trip').innerHTML = '<option value="">Not connected</option>';
    $('run').disabled = true;
    fail(error.message);
  }
}

function bounds() {
  let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
  const add = (point) => {
    if (!point || point.length < 2) return;
    const lat = point[0], lon = point[1];
    if (!isFinite(lat) || !isFinite(lon)) return;
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
  };
  for (const point of data.gps || []) add(point);
  for (const segment of data.predicted_segments || []) {
    if (Array.isArray(segment) && segment.length === 2) { add(segment[0]); add(segment[1]); }
  }
  if (!isFinite(minLat) || !isFinite(minLon)) return null;
  return { minLat, maxLat, minLon, maxLon };
}

function fit() {
  if (!data) return;
  const box = bounds();
  const width = canvas.clientWidth || 800;
  const height = canvas.clientHeight || 500;
  if (!box) { view = null; draw(); return; }
  const lat0 = (box.minLat + box.maxLat) / 2;
  const cos = Math.cos(lat0 * Math.PI / 180) || 1e-6;
  const pad = 28;
  const spanX = Math.max((box.maxLon - box.minLon) * cos, 1e-6);
  const spanY = Math.max(box.maxLat - box.minLat, 1e-6);
  const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY);
  view = {
    width, height, scale, cos,
    lon0: (box.minLon + box.maxLon) / 2,
    lat0: lat0,
  };
  draw();
}

function xy(lat, lon) {
  return [
    view.width / 2 + (lon - view.lon0) * view.cos * view.scale,
    view.height / 2 - (lat - view.lat0) * view.scale,
  ];
}

function draw() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth || 800;
  const height = canvas.clientHeight || 500;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!view || !data) return;

  if ($('showPred').checked) {
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2.4;
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    for (const segment of data.predicted_segments || []) {
      if (!Array.isArray(segment) || segment.length !== 2) continue;
      const a = xy(segment[0][0], segment[0][1]);
      const b = xy(segment[1][0], segment[1][1]);
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if ($('showGps').checked && (data.gps || []).length > 1) {
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.6;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    let first = true;
    for (const point of data.gps) {
      const p = xy(point[0], point[1]);
      if (first) { ctx.moveTo(p[0], p[1]); first = false; } else { ctx.lineTo(p[0], p[1]); }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  const gps = data.gps || [];
  if (gps.length) {
    const marker = (point, color, label, align) => {
      const p = xy(point[0], point[1]);
      ctx.beginPath();
      ctx.arc(p[0], p[1], 6.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#f8fbff';
      ctx.stroke();
      ctx.font = '600 10px ui-monospace, monospace';
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(6,10,18,.85)';
      ctx.strokeText(label, p[0] + (align === 'right' ? -11 : 11), p[1]);
      ctx.fillStyle = '#e6eefc';
      ctx.fillText(label, p[0] + (align === 'right' ? -11 : 11), p[1]);
    };
    marker(gps[0], '#22c55e', 'START', 'left');
    if (gps.length > 1) marker(gps[gps.length - 1], '#ef4444', 'END', 'right');
  }
}

function fmt(n) { return Number.isFinite(n) ? n.toLocaleString() : '—'; }
function fmtMs(ms) { return Number.isFinite(ms) ? (ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(2) + ' s') : '—'; }

async function reconstruct() {
  const tripId = $('trip').value;
  if (!tripId) return;
  clearError();
  $('loading').hidden = false;
  $('loadingText').textContent = 'Reconstructing trip ' + tripId + '…';
  $('run').disabled = true;
  status('Running candidate scoring + Viterbi…', '');
  const started = performance.now();
  try {
    const payload = await getJson('api/predict/' + encodeURIComponent(tripId));
    data = payload;
    const decoded = Array.isArray(payload.decoded_arc_ids) ? payload.decoded_arc_ids : [];
    const roundTrip = performance.now() - started;
    const server = Number.isFinite(payload.elapsed_ms) ? payload.elapsed_ms : null;
    $('mTrip').textContent = payload.trip_id;
    $('mObs').textContent = fmt(payload.observations);
    $('mDecoded').textContent = fmt(decoded.length);
    $('mArcs').textContent = fmt(new Set(decoded).size);
    $('mTime').textContent = (server === null ? '' : fmtMs(server) + ' · ') + fmtMs(roundTrip) + ' round-trip';
    $('empty').hidden = true;
    $('fit').disabled = false;
    fit();
    status('Trip ' + tripId + ' · ' + fmt(payload.observations) + ' observations · ' +
           fmt(decoded.length) + ' decoded', 'ok');
  } catch (error) {
    fail(error.message);
  } finally {
    $('loading').hidden = true;
    $('run').disabled = false;
  }
}

$('run').addEventListener('click', reconstruct);
$('fit').addEventListener('click', fit);
$('showGps').addEventListener('change', draw);
$('showPred').addEventListener('change', draw);
$('clear').addEventListener('click', () => {
  data = null;
  view = null;
  clearError();
  $('empty').hidden = false;
  $('fit').disabled = true;
  for (const id of ['mTrip', 'mObs', 'mDecoded', 'mArcs', 'mTime']) $(id).textContent = '—';
  draw();
  status('Map cleared.', '');
});
$('trip').addEventListener('change', () => {
  clearError();
  data = null;
  view = null;
  $('empty').hidden = false;
  $('fit').disabled = true;
  draw();
  const id = $('trip').value;
  status(id ? 'Trip ' + id + ' selected — run Reconstruct route.' : 'Choose a trip.', '');
});
window.addEventListener('resize', () => { if (data) fit(); else draw(); });

connect();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="PathSnap viewer that proxies the PathSnap API.")
    parser.add_argument(
        "--api",
        default=os.environ.get("PATHSNAP_API", BACKEND_URL),
        help="backend base URL, e.g. http://192.168.1.42:8000",
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface for this viewer")
    parser.add_argument("--port", type=int, default=5500, help="port for this viewer")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    backend = args.api.strip().rstrip("/")
    if not backend.startswith(("http://", "https://")):
        backend = "http://" + backend
    Handler.backend = backend

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"PathSnap viewer  ->  {url}")
    print(f"Proxying API     ->  {backend}")
    print("Stop with Ctrl+C")

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as error:
        print(f"Could not listen on {args.host}:{args.port}: {error}", file=sys.stderr)
        return 1

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - headless machines
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
