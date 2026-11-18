'use strict';
const $ = id => document.getElementById(id);
let prediction = null;
const canvas = $('map');
const ctx = canvas.getContext('2d');
const status = message => { $('status').textContent = message; };
const backend = () => $('api').value.trim().replace(/\/+$/, '');

async function request(path) {
  const base = backend();
  if (!/^https?:\/\/[^\s/]+(?::\d+)?(?:\/.*)?$/i.test(base)) throw new Error('Enter a valid http(s) backend URL');
  const response = await fetch(`${base}${path}`, { signal: AbortSignal.timeout(120000) });
  let data;
  try { data = await response.json(); } catch { throw new Error(`Invalid JSON response (HTTP ${response.status})`); }
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

function setBusy(busy) {
  $('connect').disabled = busy;
  $('predict').disabled = busy || !$('trip').value;
}

async function connect() {
  setBusy(true);
  status('Connecting to backend…');
  $('connection').textContent = 'Connecting…';
  $('connection').classList.remove('online');
  try {
    const [health, result] = await Promise.all([request('/health'), request('/trips')]);
    if (health.status !== 'ok') throw new Error('Backend health check failed');
    const trips = Array.isArray(result.trips) ? result.trips : [];
    $('trip').replaceChildren();
    for (const id of trips) {
      const option = document.createElement('option');
      option.value = String(id);
      option.textContent = String(id);
      $('trip').appendChild(option);
    }
    $('connection').textContent = health.model_ready ? 'Backend connected' : 'Model unavailable';
    $('connection').classList.add('online');
    status(`Connected · ${result.partition || 'unknown split'} · ${trips.length} available trips${health.model_ready ? '' : ' · model not ready'}`);
  } catch (err) {
    $('trip').replaceChildren();
    $('connection').textContent = 'Connection failed';
    status(`Connection error: ${err.message}`);
  } finally { setBusy(false); }
}

async function predict() {
  const id = $('trip').value;
  if (!id) return status('Select a trip first.');
  setBusy(true);
  status(`Reconstructing trip ${id}…`);
  try {
    const result = await request(`/predict/${encodeURIComponent(id)}`);
    if (!Array.isArray(result.gps) || !Array.isArray(result.predicted_segments)) throw new Error('Unexpected API response');
    prediction = result;
    $('trip-id').textContent = result.trip_id;
    $('observations').textContent = Number(result.observations).toLocaleString();
    $('matches').textContent = (result.decoded_arc_ids || []).length.toLocaleString();
    draw();
    status(`Completed trip ${id}: ${result.observations} GPS observations.`);
  } catch (err) { status(`Prediction error: ${err.message}`); }
  finally { setBusy(false); }
}

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#f8fafc'; ctx.fillRect(0, 0, w, h);
  if (!prediction) {
    ctx.fillStyle = '#64748b'; ctx.font = '24px system-ui';
    ctx.textAlign = 'center'; ctx.fillText('Select a trip and reconstruct its route', w / 2, h / 2);
    return;
  }
  const gps = prediction.gps || [], segments = prediction.predicted_segments || [];
  const points = gps.concat(segments.flatMap(segment => Array.isArray(segment) ? segment : []));
  const valid = points.filter(p => Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (!valid.length) return;
  let minLat=Infinity,maxLat=-Infinity,minLon=Infinity,maxLon=-Infinity;
  for (const [lat,lon] of valid) { minLat=Math.min(minLat,lat);maxLat=Math.max(maxLat,lat);minLon=Math.min(minLon,lon);maxLon=Math.max(maxLon,lon); }
  // Correct longitude's visual scale approximately for the latitude.
  const lonFactor = Math.max(.05, Math.cos((minLat+maxLat)/2*Math.PI/180));
  const dx = Math.max((maxLon-minLon)*lonFactor, 1e-10), dy = Math.max(maxLat-minLat, 1e-10);
  const scale = Math.min((w-90)/dx, (h-90)/dy);
  const x0=(w-dx*scale)/2, y0=(h-dy*scale)/2;
  const xy = ([lat,lon]) => [x0+(lon-minLon)*lonFactor*scale, h-y0-(lat-minLat)*scale];
  if ($('show-route').checked) {
    ctx.strokeStyle='#16a34a';ctx.lineWidth=2.5;ctx.globalAlpha=.85;
    for (const segment of segments) {
      if (!Array.isArray(segment) || segment.length!==2) continue;
      const [a,b]=segment;
      if (![...a,...b].every(Number.isFinite)) continue;
      const [ax,ay]=xy(a),[bx,by]=xy(b);
      ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);ctx.stroke();
    }
    ctx.globalAlpha=1;
  }
  if ($('show-gps').checked) {
    ctx.fillStyle='#2563eb';ctx.globalAlpha=.7;
    for (const point of gps) { if (!point.every(Number.isFinite)) continue; const [x,y]=xy(point);ctx.beginPath();ctx.arc(x,y,2,0,Math.PI*2);ctx.fill(); }
    ctx.globalAlpha=1;
  }
  if (gps.length) {
    for (const [point,color] of [[gps[0],'#22c55e'],[gps[gps.length-1],'#ef4444']]) {
      const [x,y]=xy(point);ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
    }
  }
}
$('connect').addEventListener('click', connect);
$('predict').addEventListener('click', predict);
$('trip').addEventListener('change', () => { $('predict').disabled = !$('trip').value; });
$('show-gps').addEventListener('change', draw);
$('show-route').addEventListener('change', draw);
$('reset').addEventListener('click', draw);
draw();
