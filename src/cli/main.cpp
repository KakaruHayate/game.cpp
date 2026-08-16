// Command-line entry point for game_ggml.
//
// Subcommands:
//   --version / --help                  (task 1)
//   inspect <gguf>                      (task 2)
//   extract <wav> -m <gguf> [options]   (task 10)

#include "game_ggml/game_ggml.h"
#include "game_ggml/version.h"
#include "game_ggml/config.h"
#include "game_ggml/model.h"

#include "../backend.h"
#include "../gguf_io.h"
#include "../model_impl.h"
#include "../rng.h"
#include "wav_io.h"
#include "slicer.h"
#include "midi_writer.h"
#include "text_writer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
    #include <io.h>   // _setmode
    #include <fcntl.h> // _O_BINARY
#endif

namespace fs = std::filesystem;

namespace {

// ---- tiny argument parser helpers ---------------------------------------

struct ArgView {
    int argc;
    char ** argv;
    int cursor = 0;

    bool has_next() const { return cursor < argc; }
    std::string next() { return argv[cursor++]; }
    std::string peek() const { return cursor < argc ? argv[cursor] : std::string{}; }

    std::string consume_or(const std::string & flag, const std::string & fallback) {
        if (peek() == flag && cursor + 1 < argc) {
            cursor += 2;
            return argv[cursor - 1];
        }
        return fallback;
    }
};

std::set<std::string> split_set(const std::string & s, char sep = ',') {
    std::set<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) {
            if (!cur.empty()) out.insert(cur);
            cur.clear();
        } else cur += c;
    }
    if (!cur.empty()) out.insert(cur);
    return out;
}

// ---- --version ----------------------------------------------------------

void print_usage(const char * argv0) {
    std::fprintf(stdout,
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  --version                              Print library + ggml version\n"
        "  --help                                 Show this message\n"
        "  inspect <gguf>                         Show GGUF metadata + tensor summary\n"
        "  extract <wav> -m <gguf> [options]      Transcribe a WAV file to MIDI/TXT/CSV\n"
        "  serve <gguf>                           Long-lived stdin/stdout server for a host\n"
        "\n"
        "Extract options:\n"
        "  -m, --model <path>                     Path to .gguf model                     (required)\n"
        "  -l, --language <id>                    Language id (0 = unknown)                (default: 0)\n"
        "  --output-formats <mid,txt,csv>         Comma-separated output formats            (default: mid)\n"
        "  --output-dir <dir>                     Directory for outputs                     (default: alongside input)\n"
        "  --tempo <bpm>                          MIDI tempo                               (default: 120)\n"
        "  --seg-threshold <float>                Boundary decoding threshold              (default: 0.2)\n"
        "  --seg-radius <frames>                  Boundary decoding radius                 (default: 2)\n"
        "  --est-threshold <float>                Note presence threshold                  (default: 0.2)\n"
        "  --t0 <float>                           D3PM initial t                           (default: 0.0)\n"
        "  --nsteps <int>                         D3PM sampling steps                      (default: 1)\n"
        "  --seed <uint64>                        RNG seed (0 = random_device)             (default: 0)\n"
        "  --no-slice                             Feed the full WAV as one chunk           (default: false)\n"
        "  --pitch-format name|number             Text output pitch format                 (default: name)\n"
        "  --round-pitch                          Round pitch to integer in text output    (default: false)\n"
        "  --cache-threshold <float>              DBCache normalized-L1 threshold           (default: 0.25)\n"
        "  --cache-fn-blocks <int>                DBCache front blocks per step              (default: 1)\n"
        "  --cache-warmup <int>                   D3PM steps before caching starts           (default: 1)\n"
        "  --rng-replay <path>                    Feed float32 uniform samples from file    (parity vs PyTorch)\n",
        argv0);
}

void print_version() {
    std::fprintf(stdout, "game_ggml %s · ggml %s · backends: [",
        game_ggml::version_string(), game_ggml::ggml_version_string());
    const int n = game_ggml::available_backends_count();
    const char * const * names = game_ggml::available_backends();
    for (int i = 0; i < n; ++i) {
        std::fprintf(stdout, "%s%s", names[i], i + 1 < n ? ", " : "");
    }
    std::fprintf(stdout, "]\n");
    try {
        auto * b = game_ggml::internal::init_best_backend();
        std::fprintf(stdout, "active backend: %s\n", game_ggml::internal::backend_name(b));
        game_ggml::internal::free_backend(b);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "warning: %s\n", e.what());
    }
}

// ---- inspect ------------------------------------------------------------

int cmd_inspect(int argc, char ** argv) {
    if (argc < 1) { std::fprintf(stderr, "usage: inspect <path.gguf>\n"); return 1; }
    std::string path = argv[0];
    int tensor_limit = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-n" || a == "--tensor-limit") && i + 1 < argc) {
            tensor_limit = std::atoi(argv[++i]);
        }
    }

    auto file = game_ggml::internal::GgufFile::open(path);
    auto cfg  = game_ggml::internal::load_config(file);
    std::fprintf(stdout, "== %s ==\n", path.c_str());
    std::fprintf(stdout, "  architecture : %s\n", cfg.architecture.c_str());
    std::fprintf(stdout, "  name         : %s\n", cfg.name.c_str());
    std::fprintf(stdout, "  embedding_dim: %d\n", cfg.embedding_dim);
    std::fprintf(stdout, "  encoder      : %d layers, %d heads\n", cfg.encoder.num_layers, cfg.encoder.num_heads);
    std::fprintf(stdout, "  segmenter    : %d layers, latent@%d\n", cfg.segmenter.num_layers, cfg.segmenter.latent_layer_idx);
    std::fprintf(stdout, "  estimator    : %d layers, %s attn, R=%d\n", cfg.estimator.num_layers,
                 cfg.estimator.attn_type.c_str(), cfg.estimator.region_token_num);
    auto tensors = file.list_tensors();
    std::fprintf(stdout, "  tensors      : %zu\n", tensors.size());
    for (std::size_t i = 0; i < tensors.size() && static_cast<int>(i) < tensor_limit; ++i) {
        std::fprintf(stdout, "    %-60s size=%zu\n", tensors[i].name.c_str(), tensors[i].size_bytes);
    }
    return 0;
}

// ---- serve (long-lived stdin/stdout protocol for OpenUtau) --------------

// Binary request frame on stdin (all little-endian):
//   uint32_t magic        0x53455256 ("VRES")  — requests inference
//                            0x54495155 ("UQIT")  — quit signal
//   int32_t  language
//   uint64_t seed
//   int32_t  nsteps
//   float    seg_threshold
//   int32_t  seg_radius
//   float    est_threshold
//   uint32_t n_samples
//   float[n_samples] waveform    (mono, 44.1kHz, [-1,1])
//
// stdout responses (newline-terminated JSON):
//   {"type":"ready"}                                  after model load
//   {"type":"notes","count":N,"notes":[{"o":..,"d":..,"p":..,"v":0|1},...]}
//   {"type":"error","message":"..."}                  on failure
//
// Per-request timing & progress go to stderr.

namespace serv_proto {
constexpr std::uint32_t MAGIC_INFERENCE = 0x53455256u;  // "VRES"
constexpr std::uint32_t MAGIC_QUIT      = 0x54495155u;  // "UQIT"

#pragma pack(push, 1)
struct RequestHeader {
    std::uint32_t magic;
    std::int32_t  language;
    std::uint64_t seed;
    std::int32_t  nsteps;
    float         seg_threshold;
    std::int32_t  seg_radius;
    float         est_threshold;
    std::uint32_t n_samples;
};
#pragma pack(pop)
static_assert(sizeof(RequestHeader) == 36, "RequestHeader must be 36 bytes");
}  // namespace serv_proto

// Read exactly n bytes from a binary stream; returns false on EOF/error.
static bool read_exact(std::istream & in, char * dst, std::size_t n) {
    in.read(dst, static_cast<std::streamsize>(n));
    return static_cast<std::size_t>(in.gcount()) == n;
}

// Tiny JSON string escaper (notes shouldn't contain quotes, but be safe).
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

int cmd_serve(int argc, char ** argv) {
    using namespace game_ggml;

    // Parse: serve <gguf>
    if (argc < 1) { std::fprintf(stderr, "usage: serve <gguf>\n"); return 1; }
    const std::string model_path = argv[0];

    // Switch stdin/stdout to binary mode on Windows — std::cin text mode
    // translates \n→\r\n and stops at Ctrl-Z (0x1A), which corrupts the
    // binary request header + float32 waveform.
#ifdef _WIN32
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif

    // Use binary stdin/stdout.  Disable stdin sync for speed.
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::fprintf(stderr, "[serve] loading model: %s\n", model_path.c_str());
    // Model's default constructor is private; construct via load() directly
    // and hold by unique_ptr so the error-handling try/catch can report a
    // clean JSON envelope before rethrowing/returning.
    std::unique_ptr<Model> model;
    try {
        model = std::make_unique<Model>(Model::load(model_path));
    } catch (const std::exception & e) {
        // Emit an error envelope on stdout so the host can detect failure.
        std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n",
                    json_escape(e.what()).c_str());
        std::fflush(stdout);
        std::fflush(stderr);
        return 1;
    }
    std::fprintf(stderr, "[serve] model loaded (%s)\n",
                 model->config().architecture.c_str());

    // Announce readiness.
    std::printf("{\"type\":\"ready\"}\n");
    std::fflush(stdout);

    // Reusable request header + waveform buffer.
    serv_proto::RequestHeader hdr{};
    std::vector<float> waveform;
    std::string err_msg;

    while (true) {
        if (!read_exact(std::cin, reinterpret_cast<char *>(&hdr),
                        sizeof(hdr))) {
            // Clean EOF on stdin → exit gracefully.
            std::fprintf(stderr, "[serve] stdin closed, exiting\n");
            break;
        }
        if (hdr.magic == serv_proto::MAGIC_QUIT) {
            std::fprintf(stderr, "[serve] quit signal received\n");
            break;
        }
        if (hdr.magic != serv_proto::MAGIC_INFERENCE) {
            err_msg = "bad magic in request header";
            std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n",
                        json_escape(err_msg).c_str());
            std::fflush(stdout);
            std::fprintf(stderr, "[serve] %s\n", err_msg.c_str());
            break;
        }
        if (hdr.n_samples == 0 || hdr.n_samples > 100u * 1024u * 1024u) {
            err_msg = "n_samples out of range";
            std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n",
                        json_escape(err_msg).c_str());
            std::fflush(stdout);
            continue;
        }

        // Read the waveform payload.
        waveform.resize(hdr.n_samples);
        if (!read_exact(std::cin, reinterpret_cast<char *>(waveform.data()),
                        waveform.size() * sizeof(float))) {
            err_msg = "truncated waveform payload";
            std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n",
                        json_escape(err_msg).c_str());
            std::fflush(stdout);
            break;
        }

        // Build InferParams from the header.
        InferParams p;
        p.language           = hdr.language;
        p.seed               = hdr.seed;
        p.d3pm_nsteps        = hdr.nsteps > 0 ? hdr.nsteps : 1;
        p.boundary_threshold = hdr.seg_threshold;
        p.boundary_radius    = hdr.seg_radius;
        p.note_threshold     = hdr.est_threshold;

        InferResult r;
        try {
            auto t0 = std::chrono::steady_clock::now();
            r = model->infer(waveform.data(), waveform.size(), p);
            auto dt = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr,
                "[serve] inferred %zu notes from %u samples in %.3fs\n",
                r.notes.size(), hdr.n_samples, dt);
        } catch (const std::exception & e) {
            std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n",
                        json_escape(e.what()).c_str());
            std::fflush(stdout);
            std::fprintf(stderr, "[serve] inference error: %s\n", e.what());
            continue;
        }

        // Emit notes as a single-line JSON object.
        // Compact field names (o/d/p/v) keep the payload small for long clips.
        std::string out;
        out.reserve(r.notes.size() * 32 + 64);
        out += "{\"type\":\"notes\",\"count\":";
        out += std::to_string(r.notes.size());
        out += ",\"notes\":[";
        for (std::size_t i = 0; i < r.notes.size(); ++i) {
            const auto & n = r.notes[i];
            if (i) out += ',';
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                "{\"o\":%.6f,\"d\":%.6f,\"p\":%.6f,\"v\":%d}",
                static_cast<double>(n.offset_seconds),
                static_cast<double>(n.duration_seconds),
                static_cast<double>(n.pitch_midi),
                n.voiced ? 1 : 0);
            out += buf;
        }
        out += "]}\n";
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::fflush(stdout);
    }

    std::fflush(stderr);
    return 0;
}

// ---- extract ------------------------------------------------------------

int cmd_extract(int argc, char ** argv) {
    using namespace game_ggml;
    using namespace game_ggml::cli;

    if (argc < 1) { std::fprintf(stderr, "usage: extract <wav> -m <gguf> [options]\n"); return 1; }
    const std::string input = argv[0];

    std::string model_path;
    std::string output_dir;
    std::set<std::string> output_formats = {"mid"};
    int  language     = 0;
    int  tempo        = 120;
    float seg_thr     = 0.2f;
    int   seg_radius  = 2;
    float est_thr     = 0.2f;
    float t0          = 0.0f;
    int   nsteps      = 1;
    std::uint64_t seed = 0;
    bool no_slice = false;
    std::string pitch_format = "name";
    bool round_pitch = false;
    std::string rng_replay_path;
    float db_cache_threshold = 0.25f;   // on by default for multi-step D3PM
    int   db_cache_fn_blocks = 1;
    int   db_cache_warmup    = 1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const std::string & flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s requires a value\n", flag.c_str()); std::exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")          model_path = next(a);
        else if (a == "-l" || a == "--language")       language = std::atoi(next(a).c_str());
        else if (a == "--output-formats")              output_formats = split_set(next(a));
        else if (a == "--output-dir")                  output_dir = next(a);
        else if (a == "--tempo")                       tempo      = std::atoi(next(a).c_str());
        else if (a == "--seg-threshold")               seg_thr    = std::stof(next(a));
        else if (a == "--seg-radius")                  seg_radius = std::atoi(next(a).c_str());
        else if (a == "--est-threshold")               est_thr    = std::stof(next(a));
        else if (a == "--t0")                          t0 = std::stof(next(a));
        else if (a == "--nsteps")                      nsteps = std::atoi(next(a).c_str());
        else if (a == "--seed")                        seed = std::strtoull(next(a).c_str(), nullptr, 10);
        else if (a == "--no-slice")                    no_slice = true;
        else if (a == "--pitch-format")                pitch_format = next(a);
        else if (a == "--round-pitch")                 round_pitch = true;
        else if (a == "--rng-replay")                  rng_replay_path = next(a);
        else if (a == "--cache-threshold")             db_cache_threshold = std::stof(next(a));
        else if (a == "--cache-fn-blocks")             db_cache_fn_blocks = std::atoi(next(a).c_str());
        else if (a == "--cache-warmup")                db_cache_warmup    = std::atoi(next(a).c_str());
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    if (model_path.empty()) { std::fprintf(stderr, "error: -m/--model is required\n"); return 1; }

    // Load model.
    std::fprintf(stderr, "loading model: %s\n", model_path.c_str());
    auto model = Model::load(model_path);

    // Load WAV.
    std::fprintf(stderr, "loading wav: %s\n", input.c_str());
    auto wav = load_wav_mono_f32(input, model.config().inference.audio_sample_rate);

    // Slice + inference per chunk.
    SlicerConfig slc_cfg;
    slc_cfg.sample_rate = model.config().inference.audio_sample_rate;
    std::vector<SliceChunk> chunks;
    if (no_slice) {
        chunks.push_back({0.0, wav.samples});
        std::fprintf(stderr, "slicing disabled; using 1 chunk\n");
    } else {
        chunks = slice_waveform(wav.samples.data(), wav.samples.size(), slc_cfg);
        std::fprintf(stderr, "sliced into %zu chunk(s)\n", chunks.size());
    }

    InferParams p;
    p.language            = language;
    p.boundary_threshold  = seg_thr;
    p.boundary_radius     = seg_radius;
    p.note_threshold      = est_thr;
    p.d3pm_t0             = t0;
    p.d3pm_nsteps         = nsteps;
    p.seed                = seed;
    p.db_cache_threshold  = db_cache_threshold;
    p.db_cache_fn_blocks  = db_cache_fn_blocks;
    p.db_cache_warmup     = db_cache_warmup;

    std::vector<Note> all_notes;

    // Optional bit-exact RNG replay.  When a file is provided we load the
    // whole sequence at once and reuse a single InjectedRng across every
    // chunk — matches the sequential consumption order used by PyTorch when
    // run with batch-size=1.
    std::unique_ptr<game_ggml::internal::IRandomSource> replay_rng;
    if (!rng_replay_path.empty()) {
        std::ifstream f(rng_replay_path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "error: cannot open rng file: %s\n", rng_replay_path.c_str()); return 1; }
        f.seekg(0, std::ios::end);
        const std::size_t bytes = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<float> vals(bytes / sizeof(float));
        f.read(reinterpret_cast<char*>(vals.data()), bytes);
        std::fprintf(stderr, "[rng-replay] loaded %zu uniform samples from %s\n",
                     vals.size(), rng_replay_path.c_str());
        replay_rng = std::make_unique<game_ggml::internal::InjectedRng>(std::move(vals));
    }

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        auto & ch = chunks[i];
        std::fprintf(stderr, "  chunk %zu/%zu offset=%.3fs len=%.3fs\n",
                     i + 1, chunks.size(), ch.offset_seconds,
                     double(ch.waveform.size()) / slc_cfg.sample_rate);
        InferResult r;
        if (replay_rng) {
            r = model.internals().infer_with_rng(
                ch.waveform.data(), ch.waveform.size(), p, *replay_rng);
        } else {
            r = model.infer(ch.waveform.data(), ch.waveform.size(), p);
        }
        for (auto & n : r.notes) {
            n.offset_seconds += static_cast<float>(ch.offset_seconds);
            all_notes.push_back(n);
        }
    }
    std::fprintf(stderr, "total notes: %zu\n", all_notes.size());

    // Output.
    fs::path in_path(input);
    fs::path out_dir = output_dir.empty() ? in_path.parent_path() : fs::path(output_dir);
    fs::create_directories(out_dir);
    const std::string stem = in_path.stem().string();

    TextWriteOptions text_opts;
    text_opts.round_pitch = round_pitch;
    text_opts.use_names   = (pitch_format != "number");

    if (output_formats.count("mid")) {
        MidiWriteOptions mopts;
        mopts.tempo_bpm = tempo;
        auto p_out = out_dir / (stem + ".mid");
        write_midi_file(p_out.string(), all_notes, mopts);
        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
    }
    if (output_formats.count("txt")) {
        auto p_out = out_dir / (stem + ".txt");
        write_text_file(p_out.string(), all_notes, TextFormat::Txt, text_opts);
        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
    }
    if (output_formats.count("csv")) {
        auto p_out = out_dir / (stem + ".csv");
        write_text_file(p_out.string(), all_notes, TextFormat::Csv, text_opts);
        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "--version" || cmd == "-v") { print_version(); return 0; }
        if (cmd == "--help"    || cmd == "-h") { print_usage(argv[0]); return 0; }
        if (cmd == "inspect")                   return cmd_inspect(argc - 2, argv + 2);
        if (cmd == "extract")                   return cmd_extract(argc - 2, argv + 2);
        if (cmd == "serve")                     return cmd_serve(argc - 2, argv + 2);
        std::fprintf(stderr, "error: unknown command '%s'\n\n", cmd.c_str());
        print_usage(argv[0]);
        return 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
