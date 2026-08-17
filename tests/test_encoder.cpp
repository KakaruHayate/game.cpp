// Full-encoder E2E parity test against a PyTorch reference (task 5).
//
// Walks through the encoder stage by stage (input_proj, each EBF block,
// output_norm, output_proj) and compares against PyTorch intermediates so
// bugs can be localized to the first diverging stage.

#include <gtest/gtest.h>

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include "support/ggml_test_env.h"
#include "support/reference_io.h"
#include "../src/backend.h"
#include "../src/gguf_io.h"
#include "../src/tensor_utils.h"
#include "../src/model_encoder.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace gi = game_ggml::internal;
using namespace game_ggml::test;

namespace {

class Encoder : public ::testing::Test {
protected:
    void SetUp() override {
        if (!fs::exists(ref_data_path("encoder", "enc_x_projected"))) {
            GTEST_SKIP() << "no encoder reference dumps; run dump_reference.py --category encoder";
        }

        const char * asset = std::getenv("GAME_GGML_TEST_ASSET");
        fs::path gguf_path = asset ? asset :
            fs::current_path() / ".." / "assets" / "game_small.gguf";
        if (!fs::exists(gguf_path)) {
            GTEST_SKIP() << "game_small.gguf missing at " << gguf_path;
        }
        gguf_path_ = gguf_path.string();
    }

    std::string gguf_path_;
};

TEST_F(Encoder, PerLayerParity) {
    // --- Load config + weights ---
    auto gguf = gi::GgufFile::open(gguf_path_);
    auto cfg  = gi::load_config(gguf);

    auto backend = game_ggml::internal::init_best_backend();
    ASSERT_NE(backend, nullptr);

    auto weights = gi::LoadedWeights::load_all(gguf, backend);
    auto enc_w = gi::bind_encoder_weights(weights, cfg.encoder, "encoder");

    // --- Reference intermediates ---
    auto x_in = load_ref(ref_data_path("encoder", "enc_x_projected"));
    ASSERT_EQ(x_in.shape.size(), 3u);
    const int64_t B = x_in.shape[0], T = x_in.shape[1], D = x_in.shape[2];
    ASSERT_EQ(D, cfg.embedding_dim);

    std::vector<std::pair<std::string, RefTensor>> stages;
    stages.emplace_back("after_input_proj", load_ref(ref_data_path("encoder", "enc_after_input_proj")));
    for (int i = 0; i < cfg.encoder.num_layers; ++i) {
        stages.emplace_back("after_layer_" + std::to_string(i),
            load_ref(ref_data_path("encoder", "enc_after_layer_" + std::to_string(i))));
    }
    if (cfg.encoder.use_out_norm) {
        stages.emplace_back("after_output_norm", load_ref(ref_data_path("encoder", "enc_after_output_norm")));
    }
    stages.emplace_back("after_output_proj",   load_ref(ref_data_path("encoder", "enc_after_output_proj")));

    // --- Build compute graph ---
    ggml_init_params ip{};
    ip.mem_size = 256 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ASSERT_NE(ctx, nullptr);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, /*grads=*/false);

    ggml_tensor * xin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, T, B);
    ggml_set_name(xin, "x_in");
    ggml_set_input(xin);

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "positions");
    ggml_set_input(pos);

    std::vector<ggml_tensor *> intermediates;
    auto outs = gi::build_encoder_graph(ctx, xin, enc_w, pos, cfg.encoder,
                                        /*mask=*/nullptr, &intermediates);
    ASSERT_EQ(intermediates.size(), stages.size());

    for (auto * t : intermediates) ggml_set_output(t);
    ggml_set_output(outs.x_seg);
    ggml_set_output(outs.x_est);
    ggml_build_forward_expand(graph, outs.x_seg);
    ggml_build_forward_expand(graph, outs.x_est);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ASSERT_TRUE(ggml_gallocr_alloc_graph(alloc, graph));

    ggml_backend_tensor_set(xin, x_in.as_f32(), 0, ggml_nbytes(xin));
    std::vector<int32_t> positions(T);
    for (int i = 0; i < T; ++i) positions[i] = i;
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));

    ASSERT_EQ(ggml_backend_graph_compute(backend, graph), GGML_STATUS_SUCCESS);

    // Compare each stage; stop reporting at the first divergence beyond
    // tolerance but still run through all so we get a full picture.
    bool any_fail = false;
    for (size_t i = 0; i < stages.size(); ++i) {
        auto * t = intermediates[i];
        std::vector<float> got(ggml_nelements(t));
        ggml_backend_tensor_get(t, got.data(), 0, got.size() * sizeof(float));
        const float err = compare_f32(stages[i].second, got, /*rtol=*/1e-3f, /*atol=*/3e-3f);
        std::fprintf(stderr, "[encoder/%-20s] max_abs_err = %.3e  (%zu elems)\n",
                     stages[i].first.c_str(), err, got.size());
        if (std::isnan(err)) any_fail = true;
    }

    // Final split check: x_seg / x_est are views of `full`, compare them
    // individually against their respective dumps.
    auto x_seg_ref = load_ref(ref_data_path("encoder", "enc_x_seg"));
    auto x_est_ref = load_ref(ref_data_path("encoder", "enc_x_est"));
    std::vector<float> got_seg(ggml_nelements(outs.x_seg));
    std::vector<float> got_est(ggml_nelements(outs.x_est));
    ggml_backend_tensor_get(outs.x_seg, got_seg.data(), 0, got_seg.size() * sizeof(float));
    ggml_backend_tensor_get(outs.x_est, got_est.data(), 0, got_est.size() * sizeof(float));
    const float err_seg = compare_f32(x_seg_ref, got_seg, 1e-3f, 3e-3f);
    const float err_est = compare_f32(x_est_ref, got_est, 1e-3f, 3e-3f);
    std::fprintf(stderr, "[encoder/x_seg split         ] max_abs_err = %.3e\n", err_seg);
    std::fprintf(stderr, "[encoder/x_est split         ] max_abs_err = %.3e\n", err_est);
    if (std::isnan(err_seg) || std::isnan(err_est)) any_fail = true;

    EXPECT_FALSE(any_fail) << "encoder intermediate(s) diverged — see per-stage errors above";

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    game_ggml::internal::free_backend(backend);
}

}  // namespace
