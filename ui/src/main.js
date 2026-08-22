'use strict';
const {
  app, BrowserWindow, ipcMain, dialog, shell, Menu
} = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawn } = require('child_process');
const AdmZip = require('adm-zip');

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
const USER_DIR = app.getPath('userData');
const OUDEP_DIR = path.join(USER_DIR, 'oudep');
const TMP_DIR = path.join(USER_DIR, 'tmp');

// ---------------------------------------------------------------------------
// Oudep import / discovery
// ---------------------------------------------------------------------------
function readConfig(oudepRoot) {
  // config.json is optional; carries samplerate/timestep/languages/loop.
  const cfgPath = path.join(oudepRoot, 'config.json');
  if (!fs.existsSync(cfgPath)) return null;
  try { return JSON.parse(fs.readFileSync(cfgPath, 'utf8')); }
  catch { return null; }
}

function readYamlLite(text) {
  // Minimal yaml-ish parser for the subset used in ouder.yaml (id/version/path).
  const out = {};
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z][\w.-]*)\s*:\s*(.*?)\s*$/);
    if (m) out[m[1]] = m[2].replace(/^["']|["']$/g, '');
  }
  return out;
}

function discoverModel(oudepRoot) {
  // CLI exe at root (windows .exe, unix no ext) + a single .gguf model.
  const entries = fs.readdirSync(oudepRoot, { withFileTypes: true });
  let cli = null;
  const gguFs = [];
  for (const e of entries) {
    if (!e.isFile()) continue;
    if (/^game_ggml_cli(\.exe)?$/i.test(e.name)) cli = path.join(oudepRoot, e.name);
    if (/\.gguf$/i.test(e.name)) gguFs.push(path.join(oudepRoot, e.name));
  }
  return { cli, models: gguFs.sort((a, b) => fs.statSync(b).size - fs.statSync(a).size) };
}

function listImported() {
  if (!fs.existsSync(OUDEP_DIR)) return [];
  const res = [];
  for (const d of fs.readdirSync(OUDEP_DIR, { withFileTypes: true })) {
    if (!d.isDirectory()) continue;
    const root = path.join(OUDEP_DIR, d.name);
    const dd = { id: d.name, root, cli: null, models: [], config: null, spoke: null };
    dd.config = readConfig(root);
    const disc = discoverModel(root);
    dd.cli = disc.cli;
    dd.models = disc.models;
    const metaPath = path.join(root, 'oudep.yaml');
    if (fs.existsSync(metaPath)) dd.meta = readYamlLite(fs.readFileSync(metaPath, 'utf8'));
    if (dd.cli && dd.models.length) res.push(dd);
  }
  return res;
}

function importOudep(filePath) {
  fs.mkdirSync(OUDEP_DIR, { recursive: true });
  const id = path.basename(filePath, path.extname(filePath)) + '-' + Date.now().toString(36);
  const root = path.join(OUDEP_DIR, id);
  fs.mkdirSync(root, { recursive: true });
  try {
    const zip = new AdmZip(filePath);
    zip.extractAllTo(root, true);
  } catch (err) {
    fs.rmSync(root, { recursive: true, force: true });
    throw new Error('Failed to unzip oudep: ' + err.message);
  }
  const disc = discoverModel(root);
  if (!disc.cli || !disc.models.length) {
    fs.rmSync(root, { recursive: true, force: true });
    throw new Error('Unsupported oudep: no game_ggml_cli executable or .gguf model found at package root');
  }
  // Make the CLI executable on unix.
  if (process.platform !== 'win32' && fs.existsSync(disc.cli)) {
    fs.chmodSync(disc.cli, 0o755);
  }
  return listImported().find(x => x.id === id);
}

// ---------------------------------------------------------------------------
// Audio: write Float32 mono @44100 to a 16-bit PCM WAV file
// ---------------------------------------------------------------------------
function writeWav(filePath, samples, sampleRate) {
  const n = samples.length;
  const buf = Buffer.alloc(44 + n * 2);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * 2, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);   // PCM
  buf.writeUInt16LE(1, 22);   // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buf.writeUInt16LE(2, 32);   // block align
  buf.writeUInt16LE(16, 34);  // bits
  buf.write('data', 36);
  buf.writeUInt32LE(n * 2, 40);
  for (let i = 0; i < n; i++) {
    let v = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE(Math.round(v * 32767), 44 + i * 2);
  }
  fs.writeFileSync(filePath, buf);
}

// ---------------------------------------------------------------------------
// Render: run game_ggml_cli extract on the wav -> .mid + .txt
// ---------------------------------------------------------------------------
function runCli(cli, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cli, args, {
      env: {
        ...process.env,
        GAME_GGML_THREADS: String(opts.threads || Math.max(2, os.cpus().length / 2)),
        ...(process.platform === 'win32' ? {} : {}),
      },
      windowsHide: true,
    });
    let out = ''; let err = '';
    proc.stdout.on('data', d => { out += d.toString(); });
    proc.stderr.on('data', d => { err += d.toString(); });
    proc.on('error', reject);
    proc.on('close', code => code === 0
      ? resolve({ stdout: out, stderr: err })
      : reject(new Error(`game_ggml_cli exited ${code}: ${err || out}`.slice(-2000))));
  });
}

async function renderMidi(importId, modelIdx, { wavPath, langId, tempo, nsteps, seed }) {
  const imp = listImported().find(x => x.id === importId);
  if (!imp) throw new Error('model package not found');
  const modelPath = imp.models[modelIdx];
  const outRoot = path.join(TMP_DIR, 'render-' + Date.now().toString(36));
  fs.mkdirSync(outRoot, { recursive: true });

  const args = ['extract', wavPath, '-m', modelPath,
    '--output-dir', outRoot,
    '--output-formats', 'mid,txt',
    '--tempo', String(tempo || 120),
    '--seed', String(seed ?? 42),
  ];
  if (nsteps && Number(nsteps) > 1) args.push('--nsteps', String(nsteps));
  else if (Number(nsteps) === 1) { /* default */ }
  if (langId !== undefined && langId !== null && String(langId) !== '') {
    args.push('--language', String(langId));
  }
  if (Number(nsteps) === 8) { args.push('--cache-threshold', '0.25', '--cache-fn-blocks', '1', '--cache-warmup', '1'); }

  await runCli(imp.cli, args, { threads: os.cpus().length });

  const stem = path.basename(wavPath, path.extname(wavPath));
  const mid = path.join(outRoot, stem + '.mid');
  const txt = path.join(outRoot, stem + '.txt');
  fs.rmSync(wavPath, { force: true });
  return { midPath: fs.existsSync(mid) ? mid : null, txtPath: fs.existsSync(txt) ? txt : null, outRoot };
}

// ---------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------
function registerIpc() {
  ipcMain.handle('oudep:pick-import', async () => {
    const r = await dialog.showOpenDialog({ properties: ['openFile'], filters: [{ name: 'oudep', extensions: ['oudep', 'zip'] }] });
    if (r.canceled || !r.filePaths.length) return null;
    return importOudep(r.filePaths[0]);
  });
  ipcMain.handle('oudep:list', () => listImported());
  ipcMain.handle('oudep:remove', (e, id) => {
    const root = path.join(OUDEP_DIR, id);
    if (fs.existsSync(root)) fs.rmSync(root, { recursive: true, force: true });
    return listImported();
  });
  ipcMain.handle('render:midi', (e, opts) => {
    const { importId, modelIdx, samples, sampleRate, langId, tempo, nsteps, seed } = opts;
    const wavPath = path.join(TMP_DIR, 'input-' + Date.now().toString(36) + '.wav');
    fs.mkdirSync(TMP_DIR, { recursive: true });
    writeWav(wavPath, samples, sampleRate || 44100);
    return renderMidi(importId, modelIdx, { wavPath, langId, tempo, nsteps, seed });
  });
  ipcMain.handle('export:mid', async (e, midPath) => {
    const r = await dialog.showSaveDialog({ filters: [{ name: 'MIDI', extensions: ['mid', 'midi'] }] });
    if (r.canceled || !r.filePath) return null;
    fs.copyFileSync(midPath, r.filePath);
    return r.filePath;
  });
  ipcMain.handle('shell:open', (e, target) => { if (target) shell.openPath(target); });

  // read-only file access for renderer (visualization). Path comes from the
  // main-process-owned render output dir, so it is not an arbitrary file read.
  ipcMain.handle('fs:read-text', (e, p) => fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : null);
  ipcMain.handle('fs:read-midi-notes', (e, p) => readMidiNotes(p));
}

// ---------------------------------------------------------------------------
// Minimal MIDI file parser for visualization (note-on/off -> pitch/duration)
// ---------------------------------------------------------------------------
function readMidiNotes(midPath) {
  if (!fs.existsSync(midPath)) return [];
  try {
    const b = fs.readFileSync(midPath);
    // b is a Node Buffer; but readFileSync with no encoding returns Buffer
    // whose readUInt32BE etc. work. Use Buffer everywhere.
    let pos = 0;
    const noteOn = new Map();
    let tempo = 500000; // 120 BPM default, PPQ assumption 480
    const ppq = 480;
    const notes = [];
    while (pos + 8 <= b.length && b.toString('ascii', pos, pos + 4) === 'MThd') {
      const len = b.readUInt32BE(pos + 4); pos += 8 + len;
    }
    while (pos + 8 <= b.length && b.toString('ascii', pos, pos + 4) === 'MTrk') {
      const len = b.readUInt32BE(pos + 4);
      const end = pos + 8 + len;
      let i = pos + 8;
      let tick = 0;
      while (i < end) {
        let dt = 0; let shift = 0;
        for (;;) { const k = b[i++]; dt |= (k & 0x7f) << shift; if (!(k & 0x80)) break; shift += 7; }
        tick += dt;
        const st = b[i++];
        if (st === 0xff) {          // meta
          const mtype = b[i++]; const mlen = b[i++];
          if (mtype === 0x51 && mlen === 3) tempo = b.readUIntBE(i, 3);
          i += mlen;
          continue;
        }
        if (st === 0xf0 || st === 0xf7) { while (i < end && b[i] !== 0xf7) i++; if (b[i] === 0xf7) i++; continue; }
        const type = st & 0xf0, ch = st & 0x0f;
        if (type === 0xc0 || type === 0xd0) { i += 1; continue; }   // program/chan-msg
        const d1 = b[i++], d2 = b[i++];
        if (type === 0x90 && d2 > 0) noteOn.set(ch * 128 + d1, tick);
        else if (type === 0x90 || type === 0x80) {
          const on = noteOn.get(ch * 128 + d1);
          if (on !== undefined) { notes.push({ on, off: tick, p: d1 }); noteOn.delete(ch * 128 + d1); }
        }
      }
      pos = end;
    }
    notes.sort((a, b) => a.on - b.on);
    const sec = t => t / ppq * (tempo / 1e6);
    const result = notes.map(x => ({ o: sec(x.on), d: sec(x.off) - sec(x.on), p: x.p }));
    const max = Math.max(0, ...result.map(x => x.o + x.d));
    return { notes: result, duration: sec(Math.max(0, ...notes.map(x => x.off))) || (result.length ? max : 0), tempo };
  } catch { return { notes: [], duration: 0, tempo: 500000 }; }
}



// ---------------------------------------------------------------------------
// Window / app lifecycle
// ---------------------------------------------------------------------------
function createWindow() {
  const win = new BrowserWindow({
    width: 1180,
    height: 780,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  return win;
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  registerIpc();
  fs.mkdirSync(TMP_DIR, { recursive: true });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
