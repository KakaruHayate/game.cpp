# External consumer example

Minimal standalone CMake project demonstrating third-party library integration:

```
examples/external_consumer/
├── CMakeLists.txt    # uses add_subdirectory(../..) to pull in game_ggml
└── main.cpp          # loads a WAV, runs Model::infer, prints notes
```

## Quick run

```bash
# From the repo root:
cmake -S examples/external_consumer -B /tmp/consumer
cmake --build /tmp/consumer -j
/tmp/consumer/my_app game_small.gguf /path/to/input.wav
```

## What it illustrates

1. **Clean include surface** — `main.cpp` only includes
   `<game_ggml/model.h>`, `<game_ggml/types.h>`, and `<game_ggml/errors.h>`.
   No ggml headers leak in.
2. **Alias target** — `target_link_libraries(my_app PRIVATE game_ggml::game_ggml)`
   is the recommended entry point.
3. **Custom audio I/O** — this example uses a hand-rolled PCM-16 WAV reader
   to stay dep-free; drop in `dr_wav` / `libsndfile` / `miniaudio` if you need
   broader format support.

The model's backend (CPU / Metal / CUDA / Vulkan) is inherited from the parent
`your build configuration.  To override, pass the relevant flags
when you configure:

```bash
cmake -S examples/external_consumer -B /tmp/consumer \
      -DGAME_GGML_METAL=OFF           # CPU-only build
```
