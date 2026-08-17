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

// ---- minimal glob (filename wildcards, dir part literal) ------------------
bool glob_match(const std::string & name, const std::string & pat) {
    // Iterative wildcard matcher supporting '*' and '?'; '.' literal in pat.
    std::size_t ni = 0, pi = 0, star = std::string::npos, n0 = 0;
    while (ni < name.size()) {
        if (pi < pat.size() && (pat[pi] == '?' || pat[pi] == name[ni])) { ++ni; ++pi; }
        else if (pi < pat.size() && pat[pi] == '*') { star = pi++; n0 = ni; }
        else if (star != std::string::npos) { pi = star + 1; ni = ++n0; }
        else return false;
    }
    while (pi < pat.size() && pat[pi] == '*') ++pi;
    return pi == pat.size();
}

std::vector<std::string> glob_files(const std::string & pattern) {
    std::vector<std::string> out;
    std::error_code ec;
    fs::path pat_path(pattern);
    fs::path dir = pat_path.parent_path();
    if (dir.empty()) dir = ".";
    std::string filt = pat_path.filename().string();
    for (fs::directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
        if (it->is_regular_file(ec) && glob_match(it->path().filename().string(), filt)) {
            out.push_back(it->path().string());
        }
    }
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
        "  --cache-threshold <float>              DBCache normalized-L1 threshold           (default: auto: CPU 0.25, GPU off)\n"
        "  --cache-fn-blocks <int>                DBCache front blocks per step              (default: 1)\n"
        "  --cache-warmup <int>                   D3PM steps before caching starts           (default: 1)\n"
        "  --cache-window-start <float>           only cache from this step fraction on      (default: 0)\n"
        "  --cache-window-end <float>             only cache up to this step fraction        (default: 1)\n"
        "  --cache-error-decay <float>            accumulate skipped-error decay (>0 enables) (default: 0 = off)\n"
        "  --cache-error-limit <float>            forced recompute when acc. error exceeds    (default: 0.5)\n"
        "  --cache-max-continuous <int>           cap consecutive cached steps (0=unlimited)  (default: 0)\n"
        "  --cache-bn-blocks <int>                always recompute last N tail blocks on hit  (default: 0)\n"
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
constexpr std::uint32_t MAGIC_BATCH     = 0x5441424Du;  // "MBAT" — batched inference

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

// Batch request header (same 36-byte frame, n_samples field repurposed as B).
// Followed by B × (uint32 n_samples + float[n_samples] waveform) blocks.
#pragma pack(push, 1)
struct BatchRequestHeader {
    std::uint32_t magic;
    std::int32_t  language;
    std::uint64_t seed;
    std::int32_t  nsteps;
    float         seg_threshold;
    std::int32_t  seg_radius;
    float         est_threshold;
    std::uint32_t B;   // item count
};
#pragma pack(pop)
static_assert(sizeof(BatchRequestHeader) == 36, "BatchRequestHeader must be 36 bytes");
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
        if (hdr.magic != serv_proto::MAGIC_INFERENCE &&
            hdr.magic != serv_proto::MAGIC_BATCH) {
            err_msg = "bad magic in request header";
            std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n",
                        json_escape(err_msg).c_str());
            std::fflush(stdout);
            std::fprintf(stderr, "[serve] %s\n", err_msg.c_str());
            break;
        }

        // ---- batched inference (MAGIC_BATCH) ----
        if (hdr.magic == serv_proto::MAGIC_BATCH) {
            const std::uint32_t B = hdr.n_samples;   // repurposed as item count
            if (B == 0 || B > 256) {
                err_msg = "batch item count out of range";
                std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n",
                            json_escape(err_msg).c_str());
                std::fflush(stdout);
                continue;
            }
            InferParams p;
            p.language           = hdr.language;
            p.seed               = hdr.seed;
            p.d3pm_nsteps        = hdr.nsteps > 0 ? hdr.nsteps : 1;
            p.boundary_threshold = hdr.seg_threshold;
            p.boundary_radius    = hdr.seg_radius;
            p.note_threshold     = hdr.est_threshold;

            std::vector<BatchItem> items;
            items.reserve(B);
            std::vector<std::vector<float>> payloads(B);
            bool truncated = false;
            for (std::uint32_t bi = 0; bi < B; ++bi) {
                std::uint32_t ns = 0;
                if (!read_exact(std::cin, reinterpret_cast<char *>(&ns), sizeof(ns)) ||
                    ns == 0 || ns > 100u * 1024u * 1024u) { truncated = true; break; }
                payloads[bi].resize(ns);
                if (!read_exact(std::cin, reinterpret_cast<char *>(payloads[bi].data()),
                                ns * sizeof(float))) { truncated = true; break; }
                items.push_back(BatchItem{ payloads[bi].data(), ns, hdr.language });
            }
            if (truncated) {
                std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n", "truncated batch payload");
                std::fflush(stdout);
                break;
            }

            BatchResult br;
            try {
                auto t0 = std::chrono::steady_clock::now();
                br = model->infer_batch(items, p);
                auto dt = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr,
                    "[serve] batch inferred %zu items in %.3fs\n", items.size(), dt);
            } catch (const std::exception & e) {
                std::printf("{\"type\":\"error\",\"message\":\"%s\"}\n",
                            json_escape(e.what()).c_str());
                std::fflush(stdout);
                std::fprintf(stderr, "[serve] batch inference error: %s\n", e.what());
                continue;
            }

            // Multi-item JSON: {"type":"notes_batch","items":[{count,notes},...]}
            std::string out;
            out.reserve(64 + br.items.size() * 32);
            out += "{\"type\":\"notes_batch\",\"items\":[";
            for (std::size_t bi = 0; bi < br.items.size(); ++bi) {
                const auto & r = br.items[bi];
                if (bi) out += ",";
                out += "{\"count\":"; out += std::to_string(r.notes.size());
                out += ",\"notes\":[";
                for (std::size_t i = 0; i < r.notes.size(); ++i) {
                    const auto & n = r.notes[i];
                    if (i) out += ",";
                    out += "{\"o\":" + std::to_string(n.offset_seconds) +
                           ",\"d\":" + std::to_string(n.duration_seconds) +
                           ",\"p\":" + std::to_string(n.pitch_midi) +
                           ",\"v\":" + (n.voiced ? "1" : "0") + "}";
                }
                out += "]}";
            }
            out += "]}";
            std::printf("%s\n", out.c_str());
            std::fflush(stdout);
            continue;
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

    if (argc < 1) { std::fprintf(stderr, "usage: extract <wav|dir> -m <gguf> [options]\n"); return 1; }
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
    float db_cache_threshold = -1.0f;    // auto: CPU 0.25, GPU 0 (EP-aware)
    int   db_cache_fn_blocks = 1;
    int   db_cache_warmup    = 1;
    float db_cache_window_start = 0.0f;
    float db_cache_window_end   = 1.0f;
    float db_cache_err_decay    = 0.0f;
    float db_cache_err_limit    = 0.5f;
    int   db_cache_max_cont     = 0;
    int   db_cache_bn_blocks    = 0;
    int   batch_size   = 4;              // slices collated into one batch (infer.py default)

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
        else if (a == "--cache-window-start")          db_cache_window_start = std::stof(next(a));
        else if (a == "--cache-window-end")            db_cache_window_end   = std::stof(next(a));
        else if (a == "--cache-error-decay")           db_cache_err_decay    = std::stof(next(a));
        else if (a == "--cache-error-limit")           db_cache_err_limit    = std::stof(next(a));
        else if (a == "--cache-max-continuous")        db_cache_max_cont     = std::atoi(next(a).c_str());
        else if (a == "--cache-bn-blocks")             db_cache_bn_blocks    = std::atoi(next(a).c_str());
        else if (a == "--batch-size")                  batch_size = std::atoi(next(a).c_str());
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 1;
        }
    }
    if (model_path.empty()) { std::fprintf(stderr, "error: -m/--model is required\n"); return 1; }

    // -------- expand input to a list of audio files (infer.py-compatible) ----
    std::vector<std::string> input_files;
    {
        std::error_code ec;
        fs::path inp(input);
        fs::file_status st = fs::status(inp, ec);
        if (fs::is_directory(st)) {
            for (fs::recursive_directory_iterator it(inp, ec), end;
                 it != end && !ec; it.increment(ec)) {
                std::string ext = it->path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (ext == ".wav" || ext == ".flac" || ext == ".mp3" || ext == ".m4a" || ext == ".ogg") {
                    input_files.push_back(it->path().string());
                }
            }
        } else if (fs::exists(st)) {
            input_files.push_back(input);
        } else {
            // Fall back to treating the argument as a glob pattern.
            for (const auto & match : glob_files(input)) input_files.push_back(match);
        }
        if (input_files.empty()) {
            std::fprintf(stderr, "error: no audio files found for input '%s'\n", input.c_str());
            return 1;
        }
        std::sort(input_files.begin(), input_files.end());
    }

    // Optional bit-exact RNG replay.  When provided we reuse a single
    // InjectedRng across every chunk of every file — matches the sequential
    // consumption order used by PyTorch with batch-size=1.
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

    // Load model once (shared across all input files).
    std::fprintf(stderr, "loading model: %s\n", model_path.c_str());
    auto model_core = Model::load(model_path);
    const int sample_rate = model_core.config().inference.audio_sample_rate;

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
    p.db_cache_window_start = db_cache_window_start;
    p.db_cache_window_end   = db_cache_window_end;
    p.db_cache_err_decay    = db_cache_err_decay;
    p.db_cache_err_limit    = db_cache_err_limit;
    p.db_cache_max_cont     = db_cache_max_cont;
    p.db_cache_bn_blocks    = db_cache_bn_blocks;

    SlicerConfig slc_cfg;
    slc_cfg.sample_rate = sample_rate;

    TextWriteOptions text_opts;
    text_opts.round_pitch = round_pitch;
    text_opts.use_names   = (pitch_format != "number");

    MidiWriteOptions mopts;
    mopts.tempo_bpm = tempo;

    // -------- batched extract (--batch-size>1): group same-length slices ----
    // Collect every (file, slice) into a flat item list, group slices by mel
    // frame count, and feed each group to infer_batch.  The fused encoder
    // + segmenter path activates when all group members share the same T;
    // otherwise infer_batch falls back to sequential (still correct).
    if (batch_size > 1) {
        struct SliceRef { int file_idx; double offset; std::vector<float> wav; };
        std::vector<SliceRef> slices;
        std::vector<fs::path> file_paths;
        for (const auto & f : input_files) {
            fs::path in_path(f);
            std::error_code ec2;
            WavFile wav;
            try { wav = load_wav_mono_f32(f, sample_rate); }
            catch (const std::exception & e) {
                std::fprintf(stderr, "  !! skip %s: %s\n", f.c_str(), e.what());
                continue;
            }
            if (wav.samples.empty()) continue;
            std::vector<SliceChunk> chunks;
            if (no_slice) chunks.push_back({0.0, wav.samples});
            else chunks = slice_waveform(wav.samples.data(), wav.samples.size(), slc_cfg);
            const int fi = static_cast<int>(file_paths.size());
            file_paths.push_back(in_path);
            for (auto & c : chunks) {
                // Skip tiny/silent leftovers the same way the sequential path
                // tolerates them (non-fatal).
                slices.push_back({fi, c.offset_seconds, std::move(c.waveform)});
            }
        }
        std::fprintf(stderr, "[batch] %zu slices across %zu files, batch_size=%d\n",
                     slices.size(), file_paths.size(), batch_size);

        // Group by frame count.
        std::map<int, std::vector<std::size_t>> groups;
        for (std::size_t i = 0; i < slices.size(); ++i) {
            int T = model_core.internals().frames_for(slices[i].wav.size());
            if (T <= 0) T = 1;
            groups[T].push_back(i);
        }
        int n_files_done = 0; std::vector<bool> file_done(file_paths.size(), false);
        for (auto & [T, idxs] : groups) {
            for (std::size_t base = 0; base < idxs.size(); base += (std::size_t)batch_size) {
                const std::size_t n = std::min((std::size_t)batch_size, idxs.size() - base);
                std::vector<BatchItem> items;
                items.reserve(n);
                std::vector<std::pair<int,double>> meta;  // (file_idx, slice offset)
                for (std::size_t k = 0; k < n; ++k) {
                    const auto & s = slices[idxs[base + k]];
                    items.push_back(BatchItem{ s.wav.data(), s.wav.size(),
                                               language > 0 ? language : 0 });
                    meta.push_back({s.file_idx, s.offset});
                }
                std::fprintf(stderr, "[batch] infer_batch T=%d n=%zu\n", T, n);
                auto br = model_core.infer_batch(items, p);
                // Reconstruct per-file note lists.
                std::vector<std::vector<Note>> per_file(file_paths.size());
                for (std::size_t k = 0; k < n && k < br.items.size(); ++k) {
                    auto & res = br.items[k];
                    auto & pf = per_file[meta[k].first];
                    for (auto & nt : res.notes) {
                        nt.offset_seconds += static_cast<float>(meta[k].second);
                        pf.push_back(nt);
                    }
                    file_done[meta[k].first] = true;
                }
                // Write once per file, when the file has a complete note list.
                for (std::size_t fi = 0; fi < per_file.size(); ++fi) {
                    if (!file_done[fi] || per_file[fi].empty()) continue;
                    const fs::path & in_path = file_paths[fi];
                    fs::path file_out_dir = output_dir.empty()
                        ? in_path.parent_path() : fs::path(output_dir);
                    fs::create_directories(file_out_dir);
                    const std::string stem = in_path.stem().string();
                    if (output_formats.count("mid")) {
                        auto p_out = file_out_dir / (stem + ".mid");
                        write_midi_file(p_out.string(), per_file[fi], mopts);
                        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
                    }
                    if (output_formats.count("txt")) {
                        auto p_out = file_out_dir / (stem + ".txt");
                        write_text_file(p_out.string(), per_file[fi], TextFormat::Txt, text_opts);
                        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
                    }
                    if (output_formats.count("csv")) {
                        auto p_out = file_out_dir / (stem + ".csv");
                        write_text_file(p_out.string(), per_file[fi], TextFormat::Csv, text_opts);
                        std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
                    }
                    ++n_files_done;
                    file_done[fi] = false;  // written
                }
            }
        }
        std::fprintf(stderr, "[batch] processed %d/%zu files\n", n_files_done, file_paths.size());
        return 0;
    }

    // -------- per-file: load → slice → run → write output beside input ----
    int n_files = 0, n_notes = 0;
    for (const auto & f : input_files) {
        fs::path in_path(f);
        ++n_files;
        std::fprintf(stderr, "[%d/%zu] loading wav: %s\n", n_files, input_files.size(), f.c_str());
        WavFile wav;
        try { wav = load_wav_mono_f32(f, sample_rate); }
        catch (const std::exception & e) {
            std::fprintf(stderr, "  !! skip %s: %s\n", f.c_str(), e.what());
            continue;
        }
        if (wav.samples.empty()) { std::fprintf(stderr, "  !! skip %s: empty\n", f.c_str()); continue; }

        std::vector<SliceChunk> chunks;
        if (no_slice) {
            chunks.push_back({0.0, wav.samples});
        } else {
            chunks = slice_waveform(wav.samples.data(), wav.samples.size(), slc_cfg);
        }
        std::fprintf(stderr, "  sliced into %zu chunk(s)\n", chunks.size());

        std::vector<Note> all_notes;
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            auto & ch = chunks[i];
            std::fprintf(stderr, "  chunk %zu/%zu offset=%.3fs len=%.3fs\n",
                         i + 1, chunks.size(), ch.offset_seconds,
                         double(ch.waveform.size()) / sample_rate);
            InferResult r;
            if (replay_rng) {
                r = model_core.internals().infer_with_rng(
                    ch.waveform.data(), ch.waveform.size(), p, *replay_rng);
            } else {
                r = model_core.infer(ch.waveform.data(), ch.waveform.size(), p);
            }
            for (auto & n : r.notes) {
                n.offset_seconds += static_cast<float>(ch.offset_seconds);
                all_notes.push_back(n);
            }
        }
        std::fprintf(stderr, "  total notes: %zu\n", all_notes.size());
        n_notes += static_cast<int>(all_notes.size());

        // Output (relative to out_dir).  Single-file input without --output-dir
        // keeps old behavior (beside the input); directory input writes under
        // out_dir, defaulting to the same dir as input.
        fs::path file_out_dir = output_dir.empty() ? in_path.parent_path() : fs::path(output_dir);
        fs::create_directories(file_out_dir);
        const std::string stem = in_path.stem().string();

        if (output_formats.count("mid")) {
            auto p_out = file_out_dir / (stem + ".mid");
            write_midi_file(p_out.string(), all_notes, mopts);
            std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
        }
        if (output_formats.count("txt")) {
            auto p_out = file_out_dir / (stem + ".txt");
            write_text_file(p_out.string(), all_notes, TextFormat::Txt, text_opts);
            std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
        }
        if (output_formats.count("csv")) {
            auto p_out = file_out_dir / (stem + ".csv");
            write_text_file(p_out.string(), all_notes, TextFormat::Csv, text_opts);
            std::fprintf(stderr, "wrote %s\n", p_out.string().c_str());
        }
    }
    std::fprintf(stderr, "processed %d/%zu files, %d notes total\n",
                 n_files, input_files.size(), n_notes);
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
