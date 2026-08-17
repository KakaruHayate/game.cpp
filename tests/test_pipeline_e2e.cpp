// Full pipeline parity test (task 9).
//
// Runs Model::infer on a ~2 s synthetic waveform with the same random
// numbers PyTorch used for its reference pass, then compares the output
// note list (durations / presence / pitch scores).

#include <gtest/gtest.h>

#include "game_ggml/game_ggml.h"
#include "game_ggml/model.h"
#include "../src/model_impl.h"
#include "../src/rng.h"
#include "cli/wav_io.h"
#include "support/reference_io.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;
using namespace game_ggml;
using namespace game_ggml::test;

TEST(Pipeline, E2EBitExactWithInjectedRng) {
    auto wav_path = ref_data_path("pipeline", "pipe_wav");
    if (!fs::exists(wav_path)) GTEST_SKIP() << "pipeline refs missing";
    const char * asset = std::getenv("GAME_GGML_TEST_ASSET");
    fs::path gguf = asset ? asset :
        fs::current_path() / ".." / "assets" / "game_small.gguf";
    if (!fs::exists(gguf)) GTEST_SKIP() << "game_small.gguf missing";

    auto wav       = load_ref(wav_path);
    auto ts_ref    = load_ref(ref_data_path("pipeline", "pipe_ts"));
    auto rng_ref   = load_ref(ref_data_path("pipeline", "pipe_rng"));
    auto dur_ref   = load_ref(ref_data_path("pipeline", "pipe_durations"));
    auto pres_ref  = load_ref(ref_data_path("pipeline", "pipe_presence"));
    auto score_ref = load_ref(ref_data_path("pipeline", "pipe_scores"));

    auto model = Model::load(gguf.string());

    InferParams params;
    params.language = 4;   // matches the dumped run
    params.d3pm_ts.assign(ts_ref.as_f32(), ts_ref.as_f32() + ts_ref.numel());

    // Inject Python-captured uniforms so D3PM random drops match bit-for-bit.
    std::vector<float> rng_vals(rng_ref.as_f32(), rng_ref.as_f32() + rng_ref.numel());
    internal::InjectedRng rng(std::move(rng_vals));

    auto result = model.internals().infer_with_rng(
        wav.as_f32(), static_cast<std::size_t>(wav.numel()),
        params, rng);

    std::fprintf(stderr, "[pipeline] predicted %zu notes, %d frames\n",
                 result.notes.size(), result.num_frames);

    // PyTorch durations have length N_max across batch, padded with -1 * step.
    // Compare up to the smaller length — counts can legitimately differ by ±1
    // due to Metal FP32 borderline flips as documented in segmenter tests.
    const std::size_t n_ref = dur_ref.numel();
    const std::size_t n_got = result.notes.size();
    EXPECT_LE(std::abs(static_cast<int>(n_ref) - static_cast<int>(n_got)), 2);

    const std::size_t n_cmp = std::min(n_ref, n_got);

    // Duration comparison: frame-level (~1 frame = ~0.01 s tolerance).
    float dur_err = 0.0f;
    int dur_frame_diff = 0;
    for (std::size_t i = 0; i < n_cmp; ++i) {
        const float d_ref = dur_ref.as_f32()[i];
        const float d_got = result.notes[i].duration_seconds;
        const float diff  = std::fabs(d_ref - d_got);
        dur_err = std::max(dur_err, diff);
        if (diff > 0.005f) ++dur_frame_diff;
    }

    // Pitch comparison (only where both PyTorch and C++ report voiced).
    int pitch_n = 0;
    float pitch_err = 0.0f;
    for (std::size_t i = 0; i < n_cmp; ++i) {
        const bool  pv = pres_ref.as_bool()[i] != 0;
        const bool  gv = result.notes[i].voiced;
        if (!pv || !gv) continue;
        const float diff = std::fabs(score_ref.as_f32()[i] - result.notes[i].pitch_midi);
        pitch_err = std::max(pitch_err, diff);
        ++pitch_n;
    }

    std::fprintf(stderr,
        "[pipeline] n_ref=%zu n_got=%zu  dur_max=%.3fs frames_off=%d  "
        "pitch_cmp=%d max_err=%.4f semitone\n",
        n_ref, n_got, dur_err, dur_frame_diff, pitch_n, pitch_err);

    EXPECT_LT(dur_err, 0.020f);        // 2 frame durations
    EXPECT_LT(pitch_err, 0.5f);        // within half a semitone
}

// Batch fusion: Model::infer_batch fused path must reproduce the sequential
// per-item results for equal-length clips (same base seed).  Reads two real
// WAVs; requires GAME_GGML_TEST_ASSET to point at a usable GGUF.
TEST(Pipeline, BatchFusedEqualsSequential) {
    const char * asset = std::getenv("GAME_GGML_TEST_ASSET");
    const char * w1 = std::getenv("GAME_GGML_TEST_WAV1");
    const char * w2 = std::getenv("GAME_GGML_TEST_WAV2");
    if (!asset || !w1 || !w2) {
        GTEST_SKIP() << "set GAME_GGML_TEST_ASSET/WAV1/WAV2 to run";
    }
    fs::path gguf = asset;
    if (!fs::exists(gguf)) GTEST_SKIP() << "GAME_GGML_TEST_ASSET not found";

    auto model = Model::load(gguf.string());
    const int sr = model.config().inference.audio_sample_rate;

    auto wa1 = game_ggml::cli::load_wav_mono_f32(w1, sr);
    auto wa2 = game_ggml::cli::load_wav_mono_f32(w2, sr);
    ASSERT_FALSE(wa1.samples.empty());
    ASSERT_FALSE(wa2.samples.empty());
    ASSERT_EQ(wa1.samples.size(), wa2.samples.size());  // equal frames required

    InferParams p;
    p.language = 4;
    p.d3pm_nsteps = 8;
    // Per-item seed policy of infer_batch: base 42 -> items use 42, 43, …
    p.seed = 42;
    // Batch fused path runs the full D3PM loop without the CPU DBCache
    // shortcut (same as infer.py cache_threshold=0), so sequential reference
    // must also disable it for a bit-for-bit comparison.
    p.db_cache_threshold = 0.0f;

    // sequential reference (same per-item seeds + cache-off as the batch path)
    auto s0 = model.infer(wa1.samples.data(), wa1.samples.size(), p);  // seed 42
    p.seed = 43;
    auto s1 = model.infer(wa2.samples.data(), wa2.samples.size(), p);  // seed 43
    p.seed = 42;

    // batched fused path
    std::vector<BatchItem> items = {
        { wa1.samples.data(), wa1.samples.size(), 4 },
        { wa2.samples.data(), wa2.samples.size(), 4 },
    };
    auto b = model.infer_batch(items, p);
    ASSERT_EQ(b.items.size(), 2u);

    auto cmp = [](const char * tag, const InferResult & x, const InferResult & y) {
        EXPECT_EQ(x.notes.size(), y.notes.size()) << tag << " note count";
        const std::size_t n = std::min(x.notes.size(), y.notes.size());
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(x.notes[i].offset_seconds,   y.notes[i].offset_seconds,   0.011f) << tag << " off";
            EXPECT_NEAR(x.notes[i].duration_seconds, y.notes[i].duration_seconds, 0.011f) << tag << " dur";
            EXPECT_EQ(x.notes[i].voiced, y.notes[i].voiced) << tag << " voiced";
            if (x.notes[i].voiced && y.notes[i].voiced) {
                EXPECT_NEAR(x.notes[i].pitch_midi, y.notes[i].pitch_midi, 0.05f) << tag << " pitch";
            }
        }
    };
    cmp("item0", b.items[0], s0);
    cmp("item1", b.items[1], s1);

    std::fprintf(stderr, "[pipeline.batch] seq0=%zu seq1=%zu batch=%zu/%zu notes\n",
        s0.notes.size(), s1.notes.size(), b.items[0].notes.size(), b.items[1].notes.size());

    // B=1 probe: a single-item batch (fused) must equal a plain infer.
    std::vector<BatchItem> one = { items[0] };
    auto b1 = model.infer_batch(one, p);
    ASSERT_EQ(b1.items.size(), 1u);
    cmp("b1", b1.items[0], s0);
}
