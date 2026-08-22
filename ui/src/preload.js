'use strict';
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('bridge', {
  // oudep
  pickImportOudep: () => ipcRenderer.invoke('oudep:pick-import'),
  listOudep: () => ipcRenderer.invoke('oudep:list'),
  removeOudep: (id) => ipcRenderer.invoke('oudep:remove', id),

  // render
  renderMidi: (opts) => ipcRenderer.invoke('render:midi', opts),

  // export / shell
  exportMidi: (midPath) => ipcRenderer.invoke('export:mid', midPath),
  openPath: (target) => ipcRenderer.invoke('shell:open', target),

  // visualization file reads (main-owned output paths only)
  readText: (p) => ipcRenderer.invoke('fs:read-text', p),
  readMidiNotes: (p) => ipcRenderer.invoke('fs:read-midi-notes', p),
});
