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
};

const $ = (id) => document.getElementById(id);
const pathBase = (p) => (p.replace(/\\/g, '/').split('/').pop() || p);

function refreshRenderReady() {
  const ok = !!(S.pkgId && S.pkg && S.pkg.models.length && S.samples && S.samples.length);
  $('btn-render').disabled = !ok;
  return ok;
}
Object.defineProperty(S, 'pkg', { get: () => S.pkgs.find(p => p.id === S.pkgId) || null });

// ---------------------------------------------------------------------------
// Oudep / model UI
// ---------------------------------------------------------------------------
function refreshPkgUI() {
  const pkg = S.pkg;
  const selModel = $('sel-model'), selLang = $('sel-lang');
  selModel.innerHTML = ''; selLang.innerHTML = '';
  if (!pkg) {
    $('pkg-status').textContent = '未导入引擎包';
    $('sel-model').disabled = $('sel-lang').disabled = true;
    refreshRenderReady(); return;
  }
  pkg.models.forEach((m, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = pathBase(m);
    selModel.appendChild(o);
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
  $('sel-model').disabled = $('sel-lang').disabled = false;
  $('pkg-status').textContent = `${pkg.id} · CLI ${(pkg.meta && pkg.meta.path) || pathBase(pkg.cli)} · ${pkg.models.length} 个模型`;
  // pkg.models[i] path base for display: keep simple
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
  src.buffer = decoded;                    // any source rate/channels
  src.connect(off.destination);            // downmix to mono + resample
  src.start(0);
  const rendered = await off.startRendering();
  const samples = rendered.getChannelData(0);
  return { samples, sampleRate: TARGET, srcRate: decoded.sampleRate,
           chans: decoded.numberOfChannels, dur: decoded.duration };
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
$('btn-import').onclick = async () => {
  const p = await window.bridge.pickImportOudep();
  if (p) { S.pkgs = await window.bridge.listOudep(); S.pkgId = p.id; S.modelIdx = 0; refreshPkgUI(); }
};
$('btn-remove-pkg').onclick = async () => {
  if (!S.pkgId) return;
  S.pkgs = await window.bridge.removeOudep(S.pkgId);
  S.pkgId = S.pkgs[0]?.id || null; S.modelIdx = 0; refreshPkgUI();
};
$('sel-model').onchange = (e) => { S.modelIdx = Number(e.target.value); refreshRenderReady(); };
$('sel-lang').onchange  = (e) => { S.langId = e.target.value; refreshRenderReady(); };

$('file-audio').onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    $('audio-info').textContent = '解码中…';
    const r = await decodeAudio(f);
    S.samples = r.samples; S.audioName = f.name;
    $('audio-info').textContent =
      `${f.name} · ${r.srcRate} Hz · ${r.chans}ch · ${r.dur.toFixed(1)}s → ${r.sampleRate} Hz mono`;
    drawWaveform(r.samples);
    refreshRenderReady();
  } catch (err) {
    $('audio-info').textContent = '解码失败: ' + err.message;
  }
};

$('btn-render').onclick = async () => {
  if (!refreshRenderReady()) return;
  const btn = $('btn-render'); btn.disabled = true;
  $('render-status').textContent = '渲染中…';
  try {
    const r = await window.bridge.renderMidi({
      importId: S.pkgId,
      modelIdx: S.modelIdx,
      samples: Array.from(S.samples),     // Float32 mono @44100
      sampleRate: 44100,
      langId: S.langId,
      tempo: Number($('in-tempo').value) || 120,
      nsteps: Number($('in-nsteps').value) || 1,
      seed: Number($('in-seed').value) || 0,
    });
    S.render = r;
    const notes = r.midPath ? await window.bridge.readMidiNotes(r.midPath) : null;
    S.viz = notes && notes.notes ? notes : { notes: [], duration: 0 };
    if (S.viz.notes.length) {
      drawPianoRoll(S.viz.notes, S.viz.duration);
      $('viz-empty').style.display = 'none';
      $('viz-title').textContent = `${S.viz.notes.length} 音符 · ${S.viz.duration.toFixed(2)}s`;
    } else {
      $('viz-empty').style.display = '';
      $('viz-title').textContent = '';
    }
    $('render-status').textContent = `完成。${r.midPath ? '已生成 .mid' : '(无 mid 输出)'}`;
    $('btn-export').disabled = !r.midPath;
    $('btn-open-folder').disabled = !r.outRoot;
  } catch (err) {
    $('render-status').textContent = '渲染失败: ' + (err && err.message ? err.message : err);
  } finally {
    btn.disabled = false; refreshRenderReady();
  }
};

$('btn-export').onclick = async () => {
  if (S.render && S.render.midPath) {
    const p = await window.bridge.exportMidi(S.render.midPath);
    if (p) $('render-status').textContent = '已导出: ' + p;
  }
};
$('btn-open-folder').onclick = () => { if (S.render) window.bridge.openPath(S.render.outRoot); };

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------
function drawWaveform(samples) {
  const cv = $('waveform'), ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  const n = samples.length;
  const step = Math.max(1, Math.floor(n / cv.width));
  for (let x = 0; x < cv.width; x++) {
    let mn = Infinity, mx = -Infinity;
    const a = x * step, b = Math.min(n, a + step);
    for (let i = a; i < b; i++) { mn = Math.min(mn, samples[i]); mx = Math.max(mx, samples[i]); }
    const y1 = cv.height / 2 + mn * cv.height / 2, y2 = cv.height / 2 + mx * cv.height / 2;
    ctx.fillStyle = '#7dd3fc'; ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
  }
}

function drawPianoRoll(notes, duration) {
  const cv = $('piano'), ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!duration || duration <= 0) duration = Math.max(...notes.map(n => n.o + n.d)) + 0.1;
  const pad = 40;
  const minP = Math.floor(Math.min(...notes.map(n => n.p)) - 2);
  const maxP = Math.ceil(Math.max(...notes.map(n => n.p)) + 2);
  const rowH = (cv.height - pad) / (maxP - minP + 1);
  const x0 = o => pad + o / duration * (cv.width - pad * 2);
  const yP = p => cv.height - (p - minP + 1) * rowH;

  // background + black-key shading
  for (let p = minP; p <= maxP; p++) {
    const isBlack = [1, 3, 6, 8, 10].includes((p + 60) % 12);
    ctx.fillStyle = isBlack ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.02)';
    ctx.fillRect(pad, yP(p), cv.width - pad * 2, rowH);
  }
  // grid lines
  ctx.strokeStyle = 'rgba(148,163,184,0.15)';
  for (let s = 0; s <= 20; s++) {
    const x = pad + s / 20 * (cv.width - pad * 2);
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, cv.height); ctx.stroke();
  }
  // notes
  for (const n of notes) {
    const x = x0(n.o), w = Math.max(3, (n.d / duration) * (cv.width - pad * 2));
    ctx.fillStyle = '#38bdf8'; ctx.globalAlpha = 0.85;
    ctx.fillRect(x, yP(n.p), w, rowH);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = '#0c4a6e'; ctx.lineWidth = 1;
    ctx.strokeRect(x, yP(n.p), w, rowH);
  }
  ctx.fillStyle = '#94a3b8'; ctx.font = '10px sans-serif';
  ctx.fillText(`${minP}`, 6, cv.height - 3);
  ctx.fillText(`${maxP}`, 6, pad + 8);
  ctx.fillText(`0s`, pad, pad - 6);
  ctx.fillText(`${duration.toFixed(1)}s`, cv.width - pad, pad - 6);
}

// init
window.addEventListener('DOMContentLoaded', async () => {
  S.pkgs = await window.bridge.listOudep();
  if (S.pkgs.length) { S.pkgId = S.pkgs[0].id; }
  refreshPkgUI();
});
