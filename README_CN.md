# GAME ggml 后端（GAME ggml backend）

> **语言：** [English](README.md) | [中文](README_CN.md)

基于 [ggml](https://github.com/ggerganov/ggml) 的 [GAME](https://github.com/openvpi/GAME) 歌声转 MIDI 模型原生 C++ 推理实现，支持 CPU、Metal（Apple Silicon 默认）、CUDA、Vulkan；是 `python infer.py extract` 的运行时零 Python 依赖替代品。

## 特性亮点

- **端到端 CLI** — WAV 输入，MIDI/TXT/CSV 输出（镜像 Python `extract`）
- **体积小** — 1.0-medium checkpoint 约 50 MB GGUF，约 50M 参数
- **启动快** — Metal binary-archive 补丁使 Apple Silicon 首次运行延迟低于 1 秒
  （补丁随生态解散收编于本仓库 `cmake/patches/ggml-metal-binary-archive.{patch,md}`）
- **第三方集成** — 干净的 PIMPL C++ API；`add_subdirectory` 后链接 `game_ggml::game_ggml`
- **逐位对齐** — 注入相同 RNG 时全流水线输出与 PyTorch 参考逐位一致

## 架构

```
waveform (44100 Hz mono)
      │
      ▼
   MelExtractor (pocketfft STFT + librosa 兼容 mel)
      │ mel [T, 80]
      ▼
   Encoder (EBFBackbone, 4 layers, dim=128)
      │ x_seg, x_est  各 [T, 128]
      ▼
   D3PM loop (默认 1 step；--nsteps 8 更高质量)
     ├─ remove_mutable_boundaries (stochastic)
     ├─ Segmenter (EBFBackbone, 8 layers; noise/time/lang embeddings)
     └─ decode_soft_boundaries (local-max)
      │ regions [T]  +  N
      ▼
   Estimator (JEBFBackbone, 4 layers; joint attention, mixed RoPE)
      │ pool_logits [N, 257]
      ▼
   Gaussian-blurred pitch decode → notes (offset, duration, pitch, voiced)
```

## 构建

```bash
cmake -S ggml_backend -B ggml_backend/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGAME_GGML_BUILD_TESTS=ON
cmake --build ggml_backend/build -j
```

选项：

| 选项 | 默认 | 含义 |
|---|---|---|
| `GAME_GGML_METAL`       | `ON` (仅 Apple) | Metal 后端 |
| `GAME_GGML_CUDA`        | `OFF` | CUDA 后端 |
| `GAME_GGML_VULKAN`      | `OFF` | Vulkan 后端 |
| `GAME_GGML_BUILD_CLI`   | `ON` | 构建 `game_ggml_cli` |
| `GAME_GGML_BUILD_TESTS` | `OFF` | 构建 GoogleTest 套件 |

## 转换 PyTorch checkpoint

```bash
pip install -r ggml_backend/scripts/requirements.txt
python ggml_backend/scripts/convert_pt_to_gguf.py \
    --model-dir GAME-pt-1.0-small \
    -o ggml_backend/assets/game_small.gguf
```

脚本读取指定目录下的 `model.pt` + `config.yaml` + `lang_map.json`，写出单个 GGUF（含全部 671 个 tensor（FP32）与 74 个 metadata KV）。

检查结果：

```bash
./ggml_backend/build/bin/game_ggml_cli inspect ggml_backend/assets/game_small.gguf
```

## 运行推理

```bash
./ggml_backend/build/bin/game_ggml_cli extract input.wav \
    -m ggml_backend/assets/game_small.gguf \
    --output-formats mid,txt,csv \
    --output-dir out/ \
    --tempo 120 \
    --seed 42
```

### CLI → `infer.py extract` 参数映射

| CLI 参数 | Python 对应 |
|---|---|
| `-m / --model`        | `-m` |
| `-l / --language`     | `-l`（数值 id——用 `inspect` 查看映射） |
| `--output-formats`    | `--output-formats` |
| `--output-dir`        | `--output-dir` |
| `--tempo`             | `--tempo` |
| `--seg-threshold`     | `--seg-threshold` |
| `--seg-radius`        | `--seg-radius`（帧） |
| `--est-threshold`     | `--est-threshold` |
| `--t0` / `--nsteps`   | `--t0` / `--nsteps`（ggml 默认 `--nsteps 1`；Python 默认 8） |
| `--seed`              | *(新增)* — 0 表示从操作系统取随机种子 |
| `--pitch-format`      | `--pitch-format` |
| `--round-pitch`       | `--round-pitch` |
| `--rng-replay <path>` | *(新增)* — 从文件重放 D3PM 随机数以与 PyTorch 逐位对齐 |

## 性能

Apple M4（macOS，16 核 Apple Silicon）实测，两侧各 3 次 `--rng-replay` 相同随机流，音符列表逐位对齐。

### 默认设置（`--nsteps 1`）

```
                          min      mean       max
PyTorch wall (s)         5.99      6.27      6.71   (MPS via Lightning)
ggml    wall (s)         3.05      3.06      3.06   (Metal, default binary)
Speedup                                    2.05 ×

PyTorch peak RSS      980.8 MB  981.2 MB  981.7 MB
ggml    peak RSS      334.1 MB  334.4 MB  334.8 MB
Memory ratio                                2.93 ×

PyTorch notes: 471  ┐
ggml    notes: 471  ├── 1-to-1 匹配，max |Δpitch| = 0.000 semitone
                     ┘
```

实时因子：**70.6×**（ggml）vs 34.4×（PyTorch）。

### 更高质量（`--nsteps 8`）

```
                          min      mean       max
PyTorch wall (s)        17.73     18.05     18.65
ggml    wall (s)         9.11      9.28      9.62
Speedup                                    1.94 ×
```

实时因子降至 23.3×（ggml）/ 11.97×（PyTorch），但分段质量略高（该片段多恢复 9 个音符）。

### 分阶段耗时（ONNX 对齐）

`GAME_GGML_PROFILE=1` 打印每块耗时（下为 `--nsteps 8`）：

```
encoder     ~0.17 s  (~16%)   waveform → x_seg/x_est  (mel + spec_proj + 4× EBF)
segmenter   ~0.79 s  (~78%)   x_seg → boundaries       (8× D3PM sampling steps)
estimator   ~0.06 s  (~ 6%)   x_est + regions → notes  (4× JEBF + joint attn)
```

Segmenter 占主导，因为 D3PM 循环 `--nsteps` 次。默认 `1` 保持低成本；`4` 或 `8` 以线性代价换取更高质量。

### DBCache（多步加速，默认开启）

`--nsteps > 1` 时默认开启跨步 DBCache（阈值 0.25 / 前置块 1 / 预热 1）：连续两步的
segmenter 前置块残差低于阈值时跳过尾部块、复用上一步的 tail delta——近无损近似
（音准漂移 ~0.2–0.3 cent，音符数不变），nsteps=8 时 segmenter 墙钟约减半。

- 调参：`--cache-threshold <float>`（0 = 关闭）、`--cache-fn-blocks <int>`、
  `--cache-warmup <int>`
- `--nsteps 1` 时即使设了阈值也会自动禁用缓存，保留零开销的单图 fused 路径

## 复现基准测试

```bash
# 1. 重采样到 44.1 kHz / mono（如未做）
python3 -c "
import librosa, soundfile as sf
y, _ = librosa.load('28.wav', sr=44100, mono=True)
sf.write('/tmp/28_44100.wav', y, 44100, subtype='PCM_16')"

# 2. 捕获 PyTorch 的 D3PM RNG 流（同时产出参考 MIDI）
python3 ggml_backend/scripts/align_demo.py /tmp/28_44100.wav \
    -m GAME-pt-1.0-small/model.pt \
    -g ggml_backend/assets/game_small.gguf \
    --cli ggml_backend/build/bin/game_ggml_cli \
    -l zh -o /tmp/align_out

# 3. 运行 3-per-side 子进程隔离基准
python3 ggml_backend/scripts/benchmark_align.py /tmp/28_44100.wav \
    -m GAME-pt-1.0-small/model.pt \
    -g ggml_backend/assets/game_small.gguf \
    --cli ggml_backend/build/bin/game_ggml_cli \
    --rng /tmp/align_out/align_rng.bin \
    -l zh -o /tmp/bench_out --runs 3
```

## 作为第三方库使用

```cmake
add_subdirectory(path/to/GAME/ggml_backend)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE game_ggml::game_ggml)
```

```cpp
// main.cpp
#include <game_ggml/model.h>
#include <vector>

int main() {
    auto model = game_ggml::Model::load("game_small.gguf");
    std::vector<float> waveform = /* ... 加载 44100 Hz mono ... */;

    game_ggml::InferParams params;
    params.language = 4;   // 来自 lang_map: { "zh": 4 }
    params.seed     = 42;

    auto result = model.infer(waveform.data(), waveform.size(), params);
    for (const auto & n : result.notes) {
        if (!n.voiced) continue;
        printf("  %.2fs + %.2fs : %.2f\n",
               n.offset_seconds, n.duration_seconds, n.pitch_midi);
    }
}
```

公共头文件 `<game_ggml/model.h>` 使用 PIMPL；消费者不会传递包含任何 ggml 头文件。见 [`examples/external_consumer/`](examples/external_consumer/) 的最小独立 CMake 工程。

## 测试

```bash
ctest --test-dir ggml_backend/build --output-on-failure
```

套件共 37 个测试，覆盖：后端初始化、GGUF I/O round-trip、每个算子（RMSNorm、Linear、LayerScale、Embedding、GLU-FFN、CgMLP、三种模式 RoPE、Attention、PAC、EBF block）、Encoder/Segmenter/Estimator 端到端 vs PyTorch 参考 dump、D3PM 8-step 注入 RNG 逐位一致（容忍 Metal FP32 漂移引起的 ≤2/100 边界翻转）、mel 频谱 vs `lib.feature.mel.StretchableMelSpectrogram`、Slicer、MIDI 写入器、TXT/CSV 文本写入器、全流水线逐位 E2E。

参考 dump 由 `python scripts/dump_reference.py --category all` 生成，gitignored——作为 CI 的一部分重新生成。

## 已知限制（v1）

- **仅 44100 Hz mono WAV** — 其他采样率抛 `InvalidWav`。重采样刻意不在范围内以保持体积小
- **仅 FP32 权重** — converter 输出 FP32 GGUF；量化是后续交付物
- **仅支持随附的 `1.0-small` 配置分支** — estimator 加载时对 `split` attention、learned pool merger、`region_token_num > 1`、`use_region_bias=true` 明确抛 `NotImplemented`
- **单次调用 batch size 1** — 与 `infer.py extract` 一致；并行流请持有多个 `Model` 实例
- **Metal FP32 精度** — 每次 matmul 约 1e-3；边界解码时相对 CPU 参考每几百帧可能翻转一帧

## 依赖

全部由 CMake 在 configure 时 FetchContent，不 vendoring。首次 configure 后源码树位于 `build/_deps/<name>-src/`。

| 依赖 | 版本 pin | 许可 | SPDX 标识 |
|---|---|---|---|
| [ggml](https://github.com/ggerganov/ggml) | `v0.11.0` tag | MIT | MIT |
| [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) | `cpp` 分支 `32424d20` | BSD-3-Clause | BSD-3-Clause |
| [dr_libs](https://github.com/mackron/dr_libs) | `master` 分支 `243e26ff` | Public Domain / MIT-0（双许可） | `Unlicense OR MIT-0` |
| [GoogleTest](https://github.com/google/googletest) | `v1.14.0` tag（仅测试） | BSD-3-Clause | BSD-3-Clause |

每个上游 LICENSE 文件在下载后保留于 `build/_deps/<name>-src/LICENSE*`。更新依赖：改 `cmake/Dependencies.cmake` 中的 `GIT_TAG` 后重新 configure。

## 许可

MIT —— 与上游 [GAME 项目](https://github.com/openvpi/GAME) 相同。再分发时也应附带上表列出的上游许可声明。
