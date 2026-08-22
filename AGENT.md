# AGENT.md — working notes for AI agents & contributors

Save future agents (and humans) from re-deriving the repo's non-obvious
constraints.  If you plan to touch the build system, the ggml dependency, or
the CPU/GPU backend wiring, **read this first**.

---

## 1. ggml version gate — stay on v0.19.0 (temporary anchor, hard rule)

The project pins **ggml `v0.19.0`** (`cmake/Dependencies.cmake`, URL + `URL_HASH`)
and builds the CI CPU package with **`GGML_NATIVE=ON`** (captures the runner
CPU — that is why that job is never cached).  This is load-bearing:

- v0.20.x turned `GGML_CPU_ALL_VARIANTS` into **dlopen MODULE plugins**
  (`GGML_BACKEND_DL`), and DL mode does **not** link the CPU backend into the
  ggml umbrella target.
- Our `src/backend.cpp` calls the CPU backend API **directly**
  (`ggml_backend_cpu_init`, `ggml_backend_cpu_set_n_threads`,
  `ggml_threadpool_new`, `ggml_backend_cpu_set_threadpool`).  With DL those
  become `undefined reference` link errors.
- `GGML_NATIVE` and `GGML_BACKEND_DL` are mutually exclusive upstream
  (ggml-cpu CMake `FATAL_ERROR`).

**Therefore:** do **not** bump the ggml tag to v0.20.x, and do **not** flip the
CI CPU job to the DL/ALL_VARIANTS portable baseline, until `backend.cpp` is
refactored to load the CPU backend via `GGML_BACKEND_DL`.  Only after that
refactor is the gate eligible for review.

When the gate eventually lifts, re-verify:
1. Both in-repo patches apply to the new tag (`git apply --check` against the
   fetched ggml source).
2. The CUDA arch list still passes ggml's CMake.
3. `src/backend.cpp` `ggml_version_string()` matches the new tag.
4. README/README_CN dependency table matches (it drifted before — see #4).

---

## 2. Direct CPU backend symbols are a linked dependency, not dlopen

`src/backend.cpp` calls ggml CPU backend API **directly**
(`ggml_backend_cpu_init`, `ggml_threadpool_new`, ...).  This couples us to a
build where the CPU backend is PUBLIC-linked into the ggml umbrella target
(non-DL).  That is the root cause of pitfall #1.  Any change that turns the
CPU backend into a dlopen module must come with the backend.cpp refactor.

---

## 3. In-repo ggml patches are load-bearing (verify before removing)

Two ggml patches live in `cmake/patches/` and are re-applied by
`cmake/Dependencies.cmake` `game_ggml_apply_patch` (idempotent; fails if the
diff stops applying):

- `ggml-metal-binary-archive.patch` — Metal first-run <1s (MTLBinaryArchive
  PSO cache).  Pairs with `GGML_METAL_EMBED_LIBRARY OFF`.
- `ggml-vulkan-pipeline-cache.patch` — Vulkan cold-start PSO persistence
  (disk-backed `VkPipelineCache`).

They are **project-owned** forks of upstream, not part of ggml main.  Do not
delete them; keep their `.md` baseline per anchor tag.

---

## 4. Doc / pin drift is a known trap

Both READMEs previously said ggml `v0.11.0` while the pin was `v0.19.0` (then
`v0.20.2`).  Always cross-check `cmake/Dependencies.cmake` against the docs;
never trust the README version row by itself.  Keep both README dependency
tables in sync with the actual pin.

---

## 5. CPU ISA: NATIVE is a headline, not the whole story

- `GGML_NATIVE` adds `-march=native` (GCC/Clang) or MSVC `FindSIMD`.
- It does NOT flip ggml's hand-written SIMD kernels — those key off
  `GGML_AVX*/GGML_AVX512*` options.  Real AVX-512 use requires enabling e.g.
  `-DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON`.
- `GGML_LLAMAFILE` (project option `GAME_GGML_LLAMAFILE`, default ON) routes
  CPU `mul_mat` through llamafile `sgemm` (tinyBLAS) for Q8_0/F32/BF16 on
  AVX2+.  Trade-off: it changes the FP summation order → CPU bit-exactness
  tests must be re-run when toggled, and the CI cpu job validates the combo.

---

## 6. CUDA arch list + toolkit coupling

The CI CUDA arch list `75;80;86;89;90;120-virtual` requires CUDA 12.8+ for
Blackwell (`sm_100`/`sm_120`) targets; `compute_120` PTX gives RTX 50-series
JIT.  Older toolkits can't compile `120-virtual` — trim the list rather than
"fixing" by touching NATIVE (the CUDA job already correctly uses
`GGML_NATIVE=OFF`; unrelated).

---

## 7. CPU/macOS cross-arch footguns

- CI `linux-x64-cpu` is **`GGML_NATIVE=ON`** (runner-ISA capture) and must
  **never** be restored from the `_deps` cache — runners rotate CPU
  generations and a stale cache ships Illegal-instruction binaries.  The
  cache step already skips this job; keep it that way.
- `macos-x64-metal` cross-compile sets `GGML_NATIVE=OFF` — otherwise the CPU
  backend detects the ARM host (`apple-m1`) and fails.
- NVCC/VS version coupling on Windows is documented in BUILDING.md; don't
  modernize blindly.

---

## 8. DBCache & FP drift — the segmenter is deliberately approximate

`--nsteps > 1` engages DBCache (cross-step tail reuse).  It is near-lossless
on purpose; the **device-side** decision metric exists specifically to avoid
host round-trips on GPU.  Changing the threshold/defaults/robustness knobs
silently changes note output — measure frame-level metrics, not just note
count, before "improving" it.  On Vulkan, Q8_0 can flip a boundary note vs
F32; bit-consistent users use F32.

---

## 9. Front-end & backend parallelism live in two places

`src/mel.cpp` uses its own small C++ thread pool + batched pocketfft; CPU
graph compute uses a persistent ggml threadpool (hybrid polling).  They are
independent.  Keep the mel pool ≤ 8 threads and guard by frame count (short
clips shouldn't spawn threads).

---

## 10. Don't fight the sandbox for writes outside the repo

The git metadata of a checkout may live in a **different directory than the
worktree** (linked worktree; `git rev-parse --git-dir` finds it).  Git index
writes (branch/stash/commit) write to that directory, not the worktree.  If a
git write fails with "Permission denied" on `index.lock`, resolve the actual
git dir with `git rev-parse --git-dir` (`--absolute-git-dir` for a full path),
and it's almost always the harness sandbox — **escalate once** with
`sandbox_permissions: danger-full-access` + justification; do not retry in a
loop.  Never hard-code a local absolute path in committed docs.

---

## 11. Scripts are part of the release pipeline

`scripts/convert_pt_to_gguf.py`, the quant config, and the
benchmark/alignment scripts back CI's `prepare-model` and reproducible
benches.  Deleting them breaks release packaging.  If a branch PR shows
script deletions, they are probably an accident unless the PR message says
otherwise.

---

## Rules of thumb

- Prefer explicit `-D...=ON/OFF` at configure over env/`CMakeCache` surgery;
  re-define a FetchContent dependency → delete `build/` and reconfigure.
- Any change to `cmake/Dependencies.cmake`, `cmake/patches/`, or CI flags
  gets reviewed against the pitfalls above before pushing.
- When in doubt, rebase a small branch onto `origin/main` and let CI (and
  CodeRabbit) review the exact delta — that is what this file is for.
