# Batch inference design (`batchinfer`)

Status: design.  Mirrors GAME `infer.py extract` batching semantics
(collate slices of several audio files → run each pipeline stage on
`[B, T]` tensors with a per-sample valid mask → split results back).

## Reference semantics (GAME `infer.py`)

From `inference/data.py` + `inference/me_infer.py`:

- Slicer produces chunks; `SlicedAudioFileIterableDataset.collate` pads chunk
  waveforms to the batch's max length with `collate_nd(..., pad_value=0.)`.
- The whole model runs on the padded `[B, T_num_samples]` waveform:
  - `forward_encoder`: mel → mask; EBF `masked_fill(~mask_unsqueeze(-1), 0)`
    at block boundaries (in attention, FFN, residuals);
  - `forward_segmenter_main`: the D3PM loop runs over `t` for the WHOLE batch
    (`ti.expand(B)`) → `(B, T_num_frames)` boundaries, no per-sample schedule;
  - `forward_estimator`: pool tokens padded to `maxN = max(N_i)`, `n_mask`
    per sample, joint attention mask built per sample over `(maxN+T)²`.
- Callbacks write one output file per input audio file.
- There is no dynamic per-sample get_rows / hidden-state branching — the
  backend just needs `[B, ...]` tensors and two masks (frame mask `[B,T]`,
  note mask `[B,Nmax]`).

## ggml shape policy

Use the tensor dimensions the codebase already treats as batch:

- feature tensor: `(D, T, B)`  (ne0=D, ne1=T, ne2=B)
- positions / noise / regions indexes: `(T, B)` int32 when per-sample
  (ne0=1, ne1=T, ne2=B), or `(T,)` shared across the batch when identical
  (position 0..T-1 is shared — RoPE positions are NOT per-sample here).
- attention layout for `ggml_flash_attn_ext`: `(D_head, H, T, B)`;
  mask layout `(kv=T, q=T, nb_head=1, nb_batch=B)` fp16 (0.0 = allowed).

Affected builders:

| stage | input | batch-sensitive part |
|---|---|---|
| encoder | mel `(D_mel,T,B)` | masks at EBF boundaries; positions shared |
| segmenter | x_seg `(D,T,B)`, noise `(T,B)`, t scalar, lang `(B,)` | noise/lang per-sample, t per-step shared |
| estimator | x_est `(D,T,B)`, regions `(T,B)`; N→Nmax | joint-attn mask per-sample, pool per-sample |

## Inference loop (host)

`infer_with_rng` generalises to `infer_batch`:

- one mel pass per audio (front-end not vectorised — cheap vs nn compute);
- one D3PM loop over `t` (shared schedule), each step runs the segmenter on
  the whole `(D,T,B)` batch at once;
- per-sample boundary/state arrays live as `[T, B]` host vectors
  (boundaries, known, mask, noise_mod, probs …);
- deterministic RNG: a *separate* MT19937 per sample (seed derivation
  documented below) so batch output == sequential per-sample output.

## API surface

```cpp
// types.h / model.h
struct BatchItem {
    const float * waveform; std::size_t n_samples;
    int language = 0;
};
struct BatchInferParams : InferParams {};
struct BatchResult {
    std::vector<InferResult> items;   // each fully decoded incl. notes
    std::vector<int> per_item_frames;
};
std::vector<InferResult> infer_batch(
    const std::vector<BatchItem> & items, const BatchInferParams & params);
```

`infer_batch` falls back to repeated `infer` when `items.size()==1` and params
don't request batching (default behavior), then the fused `[B,...]` path.

CLI: `extract` accepts a directory / glob as the input positional (like
infer.py) + `--batch-size`; chained slices of every file are collated per
recommendation.  serve: protocol gains a batch request bearing N audio blocks.

## Execution plan

1. Refactor builders to take `B` (keep B=1 bit-identical: same host buffers,
   pos arrays, same graph → same results).
2. `batch` inference of the D3PM loop body + encoder mask handling (EBF
   `masked_fill` semantics).
3. Estimator batch: Nmax padding + per-sample joint mask + n_mask.
4. CLI directory/glob + `--batch-size`; serve batch request.
5. Verification: batch(B) vs sequential(B×single) bit/quality identical;
   benchmark throughput on the 7-channel setup.

## RNG seed policy (deterministic batch == sequential)

Each batch item uses its own MT19937 seed.  When seeds are all equal the
streams diverge after the encoder (each item's rng advances independently);
to make batch == sequential, per-item seeds follow `seed + item_index` when a
base seed is provided and > 0.  In practice the CLI default (random seed) is
per-process; each batch run gets one random base and derives per-item seeds.
