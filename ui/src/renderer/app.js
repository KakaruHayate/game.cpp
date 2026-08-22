'use strict';
// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const S = {
  pkgs: [],
  pkgId: null,
  modelIdx: 0,
  langId: '',
  samples: null,
  audioName: '',
  render: null,   // { midPath, txtPath, outRoot }
  viz: null,      // { notes:[{o,d,p}], duration }
  zoom: 60,       // px/s piano-roll zoom
};
const $ = (id) => document.getElementById(id);
const pathBase = (p) => (p.replace(/\\/g, '/').split('/').pop() || p);
Object.defineProperty(S, 'pkg', { get: () => S.pkgs.find(p => p.id === S.pkgId) || null });

// status dots
function setDot(elId, state) {
  const el = $(elId); if (!el) return;
  el.className = 'dot' + (state === 'ok' ? ' ok' : state === 'bad' ? ' bad' : state === 'run' ? ' running' : '');
}
function log(msg) {
  const el = $('log'); if (!el) return;
  el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n` + el.textContent;
}

function refreshRenderReady() {
  const ok = !!(S.pkgId && S.pkg && S.pkg.models.length && S.samples && S.samples.length);
  $('btn-render').disabled = !ok;
  return ok;
}

// ---------------------------------------------------------------------------
// Oudep / model UI
// ---------------------------------------------------------------------------
function refreshPkgUI() {
  const pkg = S.pkg;
  const selModel = $('sel-model'), selLang = $('sel-lang');
  selModel.innerHTML = ''; selLang.innerHTML = '';
  if (!pkg) {
    $('pkg-status').textContent = '未导入引擎包'; setDot('modelDot', '');
    $('engine-status').textContent = '——'; setDot('engineDot', '');
    const opt = document.createElement('option'); opt.textContent = '— 未导入 —'; selModel.appendChild(opt);
    const ol = document.createElement('option'); ol.textContent = '—'; selLang.appendChild(ol);
    selModel.disabled = selLang.disabled = true;
    refreshRenderReady(); return;
  }
  pkg.models.forEach((m, i) => {
    const o = document.createElement('option'); o.value = i; o.textContent = pathBase(m); selModel.appendChild(o);
  });
  const langs = (pkg.config && pkg.config.languages) ? pkg.config.languages : {};
  const entries = Object.entries(langs);
  if (entries.length) {
    for (const [name, id] of entries) { const o = document.createElement('option'); o.value = id; o.textContent = `${name} (#${id})`; selLang.appendChild(o); }
    S.langId = entries[0][1];
  } else {
    const o = document.createElement('option'); o.value = ''; o.textContent = '默认'; selLang.appendChild(o);
    S.langId = '';
  }
  selModel.value = S.modelIdx; selLang.value = S.langId;
  selModel.disabled = selLang.disabled = false;
  $('pkg-status').textContent = `${pkg.id} · ${pathBase(pkg.cli)}`; setDot('modelDot', 'ok');
  $('engine-status').textContent = `ggml · ${pkg.models.length} 模型`; setDot('engineDot', 'ok');
  refreshRenderReady();
}

// ---------------------------------------------------------------------------
// Audio decode + resample -> Float32 mono @ 44100
// ---------------------------------------------------------------------------
async function decodeAudio(file) {
  const buf = await file.arrayBuffer();
  const ctx = new AudioContext();
  let decoded;
  try { decoded = await ctx.decodeAudioData(buf); }
  finally { ctx.close().catch(() => {}); }
  const TARGET = 44100;
  const outLen = Math.max(1, Math.round(decoded.duration * TARGET));
  const off = new OfflineAudioContext(1, outLen, TARGET);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start(0);
  const rendered = await off.startRendering();
  const samples = rendered.getChannelData(0);
  return { samples, sampleRate: TARGET, srcRate: decoded.sampleRate,
           chans: decoded.numberOfChannels, dur: decoded.duration };
}

async function loadAudioFile(f) {
  if (!f) return;
  try {
    $('audio-info').textContent = '解码中…'; setDot('audioDot', 'run'); log(`decode: ${f.name}`);
    const r = await decodeAudio(f);
    S.samples = r.samples; S.audioName = f.name;
    $('audio-info').textContent =
      `${f.name} · ${r.srcRate} Hz · ${r.chans}ch · ${r.dur.toFixed(1)}s → ${r.sampleRate} Hz mono`;
    setDot('audioDot', 'ok');
    drawWaveform(r.samples);
    refreshRenderReady();
  } catch (err) {
    $('audio-info').textContent = '解码失败: ' + err.message; setDot('audioDot', 'bad'); log('decode error: ' + err.message);
  }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
$('btn-import').onclick = async () => {
  const p = await window.bridge.pickImportOudep();
  if (p) { S.pkgs = await window.bridge.listOudep(); S.pkgId = p.id; S.modelIdx = 0; refreshPkgUI(); log(`imported: ${p.id}`); }
};
$('sel-model').onchange = (e) => { S.modelIdx = Number(e.target.value); refreshRenderReady(); };
$('sel-lang').onchange  = (e) => { S.langId = e.target.value; refreshRenderReady(); };

const dz = $('dropZone');
dz.onclick = () => $('file-audio').click();
$('file-audio').onchange = (e) => { loadAudioFile(e.target.files[0]); };
['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', (e) => {
  e.preventDefault(); dz.classList.remove('drag');
  if (e.dataTransfer.files[0]) loadAudioFile(e.dataTransfer.files[0]);
});

$('btn-render').onclick = async () => {
  if (!refreshRenderReady()) return;
  const btn = $('btn-render'); btn.disabled = true;
  $('render-status').textContent = 'transcribing…'; setDot('runDot', 'run'); log('render start');
  try {
    const r = await window.bridge.renderMidi({
      importId: S.pkgId, modelIdx: S.modelIdx,
      samples: Array.from(S.samples), sampleRate: 44100,
      langId: S.langId,
      tempo: Number($('in-tempo').value) || 120,
      nsteps: Number($('in-nsteps').value) || 1,
      seed: Number($('in-seed').value) || 0,
    });
    S.render = r;
    const notes = r.midPath ? await window.bridge.readMidiNotes(r.midPath) : null;
    S.viz = notes && notes.notes ? notes : { notes: [], duration: 0 };
    if (S.viz.notes.length) { drawRoll(S.viz.notes, S.viz.duration, S.zoom); log(`notes=${S.viz.notes.length}`); }
    else { log('render ok, 0 notes'); clearCanvas($('roll')); }
    $('render-status').textContent = `done · ${S.viz.notes.length} notes · ${r.midPath ? 'mid ✓' : '(no mid)'}`;
    $('btn-export').disabled = !r.midPath;
    $('btn-open-folder').disabled = !r.outRoot;
    setDot('runDot', S.viz.notes.length ? 'ok' : 'bad');
  } catch (err) {
    $('render-status').textContent = '渲染失败: ' + (err && err.message ? err.message : err);
    setDot('runDot', 'bad'); log('render error: ' + (err && err.message ? err.message : err));
  } finally {
    btn.disabled = false; refreshRenderReady();
  }
};
$('btn-export').onclick = async () => {
  if (S.render && S.render.midPath) {
    const p = await window.bridge.exportMidi(S.render.midPath);
    if (p) { $('render-status').textContent = '已导出: ' + p; log('export -> ' + p); }
  }
};
$('btn-open-folder').onclick = () => { if (S.render) window.bridge.openPath(S.render.outRoot); };

// ---------------------------------------------------------------------------
// Canvas: waveform + 88-key piano roll
// ---------------------------------------------------------------------------
function clearCanvas(canvas) { const c = canvas.getContext('2d'); c.clearRect(0, 0, canvas.width, canvas.height); }

function drawWaveform(samples) {
  const cv = $('waveform'), ctx = cv.getContext('2d');
  const cssW = Math.max(320, Math.floor(cv.clientWidth || 800));
  cv.width = cssW; cv.height = 140;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#f6f6f6'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = '#ddd'; ctx.beginPath(); ctx.moveTo(0, cv.height/2); ctx.lineTo(cv.width, cv.height/2); ctx.stroke();
  const n = samples.length, step = Math.max(1, Math.floor(n / cv.width));
  ctx.fillStyle = '#111';
  for (let x = 0; x < cv.width; x++) {
    let mn = Infinity, mx = -Infinity, a = x * step, b = Math.min(n, a + step);
    for (let i = a; i < b; i++) { mn = Math.min(mn, samples[i]); mx = Math.max(mx, samples[i]); }
    ctx.fillRect(x, cv.height / 2 + mn * cv.height / 2, 1, Math.max(1, (mx - mn) * cv.height / 2));
  }
}

const PITCH_LO = 21, PITCH_HI = 108;
const KEY_W = 64, RULER = 24, ROW_H = 16;
function drawRoll(notes, duration, zoom) {
  const cv = $('roll');
  if (!duration || duration <= 0) duration = Math.max(...notes.map(n => n.o + n.d)) + 0.1;
  // zoom (px/s) may be user-set via the slider; otherwise a fixed floor so
  // long audio produces a genuinely wide scrollable canvas (never compressed
  // to one screen).
  const pxPerSec = (typeof zoom === 'number' && zoom > 0) ? zoom : Math.max(24, 480 / Math.max(duration, 4));
  const totalMin = Math.min(...notes.map(n => n.p)), totalMax = Math.max(...notes.map(n => n.p));
  const lo = Math.max(PITCH_LO, totalMin - 3), hi = Math.min(PITCH_HI, totalMax + 3);
  const W = Math.max(800, Math.round(KEY_W + duration * pxPerSec));
  const H = RULER + (hi - lo + 1) * ROW_H;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const y = p => RULER + (hi - p) * ROW_H;
  const x = t => KEY_W + t * pxPerSec;

  ctx.fillStyle = '#fefefe'; ctx.fillRect(0, 0, W, H);
  for (let p = lo; p <= hi; p++) {
    const isBlack = [1, 3, 6, 8, 10].includes((p + 60) % 12);
    ctx.fillStyle = isBlack ? '#f0f0f0' : '#fafafa';
    ctx.fillRect(0, y(p), W, ROW_H);
  }
  ctx.strokeStyle = '#ececec'; ctx.lineWidth = 1;
  const tStep = Math.max(0.5, Math.round(duration / 20) * 0.5);
  for (let t = 0; t <= duration + 0.001; t += tStep) {
    ctx.beginPath(); ctx.moveTo(x(t), RULER); ctx.lineTo(x(t), H); ctx.stroke();
  }
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, RULER);
  ctx.fillStyle = '#666'; ctx.font = '10px system-ui';
  for (let t = 0; t <= duration + 0.001; t += Math.max(1, Math.round(duration / 20))) {
    ctx.fillText(`${t}s`, x(t) + 3, RULER - 6);
  }
  ctx.font = '9px ui-monospace, monospace';
  const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  for (let p = lo; p <= hi; p++) {
    const nm = names[(p + 60) % 12];
    if (nm.includes('#')) continue;
    ctx.fillStyle = '#888'; ctx.fillText(`${nm}${Math.floor(p / 12) - 1}`, 6, y(p) + ROW_H - 3);
  }
  ctx.fillStyle = '#2b7a3c';
  for (const n of notes) {
    if (n.p < lo || n.p > hi) continue;
    const nx = x(n.o), nw = Math.max(2, n.d * pxPerSec - 1);
    ctx.fillRect(nx, y(n.p), nw, ROW_H - 1);
  }
}

const zoomEl = $('zoom');
if (zoomEl) zoomEl.addEventListener('input', () => {
  const z = Number(zoomEl.value) || 60;
  S.zoom = z; const v = $('zoomVal'); if (v) v.textContent = z;
  if (S.viz && S.viz.notes.length) drawRoll(S.viz.notes, S.viz.duration, z);
});

// init
window.addEventListener('DOMContentLoaded', async () => {
  S.pkgs = await window.bridge.listOudep();
  if (S.pkgs.length) { S.pkgId = S.pkgs[0].id; }
  refreshPkgUI();
});
