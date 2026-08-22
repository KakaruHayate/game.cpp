# GAME MIDI Studio (Electron UI)

A simple cross-platform desktop UI for the GAME singing-voice-to-MIDI engine
(`game.cpp` / `game_ggml_cli`). It imports an `.oudep` package, decodes/resamples
an vocal audio file to the engine input format, renders MIDI, visualizes notes,
and exports `.mid`.

## Features

- **Import .oudep** — unpack a `game.cpp` CI-produced package
  (`game_ggml-<platform>.oudep`): discovers the `game_ggml_cli` executable, the
  `.gguf` model(s), `config.json` (samplerate / languages), and `oudep.yaml`.
- **Audio input** — any format Chromium can decode (wav / mp3 / flac / m4a /
  aac / ogg / opus / mp4 / webm…); auto **resamples to 44.1 kHz mono**
  (the engine requirement) via Web Audio, and shows a waveform.
- **Render MIDI** — runs `game_ggml_cli extract` with the imported model,
  selected language, tempo, nsteps (1 or 8) and seed; DBCache defaults for n8.
- **Visualize** — piano-roll canvas of the rendered notes (pitch vs time).
- **Export .mid** — save the rendered `.mid` anywhere; or open the output dir.

## Build & run

```bash
cd ui
npm install            # installs electron + builder + adm-zip
npm start              # run in dev
npm run dist:win       # or dist:mac / dist:linux (electron-builder)
```

CI: `.github/workflows/build-ui.yml` builds the app for Windows / macOS /
Linux, independently from the oudep-packaging `ci.yml` (triggered by
`workflow_dispatch`, or any push touching `ui/**`).

## Notes

- The engine (`game_ggml_cli`) only accepts **44.1 kHz mono 16-bit WAV**; the UI
  handles conversion. Source formats with unusual sample rates or stereo are
  downmixed/resampled automatically.
- `langId` comes from the package's `config.json` `languages` map; if absent,
  the CLI default is used.
- The `.oudep` is unpacked under the app's `userData/oudep/`; packages can be
  removed from the UI.
- `nsteps=8` enables the segmenter DBCache (default threshold 0.25) for quality,
  matching the repo's default high-quality path.
