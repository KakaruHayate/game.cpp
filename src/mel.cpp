#include "game_ggml/mel.h"
#include "game_ggml/errors.h"

#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <thread>
#include <vector>

namespace game_ggml {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// librosa Slaney mel scale (default, htk=False).
float hz_to_mel_slaney(float hz) {
    const float f_sp       = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep     = std::log(6.4f) / 27.0f;
    if (hz >= min_log_hz) return min_log_mel + std::log(hz / min_log_hz) / logstep;
    return hz / f_sp;
}

float mel_to_hz_slaney(float mel) {
    const float f_sp       = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep     = std::log(6.4f) / 27.0f;
    if (mel >= min_log_mel) return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    return mel * f_sp;
}

// [n_mels, n_fft/2 + 1] filterbank, row-major.
std::vector<float> make_mel_filterbank(int n_fft, int n_mels, int sample_rate,
                                        float fmin, float fmax) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> fb(static_cast<std::size_t>(n_mels) * n_bins, 0.0f);

    std::vector<float> fft_freqs(n_bins);
    for (int k = 0; k < n_bins; ++k) {
        fft_freqs[k] = static_cast<float>(k) * sample_rate / static_cast<float>(n_fft);
    }
    const float mel_min = hz_to_mel_slaney(fmin);
    const float mel_max = hz_to_mel_slaney(fmax);
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const float m = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        hz_points[i] = mel_to_hz_slaney(m);
    }
    for (int m = 0; m < n_mels; ++m) {
        const float lower  = hz_points[m];
        const float center = hz_points[m + 1];
        const float upper  = hz_points[m + 2];
        const float enorm  = 2.0f / (upper - lower);   // Slaney norm
        for (int k = 0; k < n_bins; ++k) {
            const float f = fft_freqs[k];
            float w = 0.0f;
            if (f >= lower && f <= center) w = (f - lower) / (center - lower);
            else if (f > center && f <= upper) w = (upper - f) / (upper - center);
            fb[m * n_bins + k] = w * enorm;
        }
    }
    return fb;
}

// Hann window matching torch.hann_window (periodic).
std::vector<float> make_hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) w[i] = 0.5f - 0.5f * std::cos(2.0f * kPi * i / n);
    return w;
}

// Reflect-pad matching torch.nn.functional.pad(mode='reflect') which
// mirrors without the edge sample: pad[i] = src[pad_left - i].
void reflect_pad(const float * src, std::size_t n, int pad_l, int pad_r,
                  std::vector<float> & dst) {
    dst.resize(n + pad_l + pad_r);
    for (int i = 0; i < pad_l; ++i) dst[i] = src[pad_l - i];
    std::memcpy(dst.data() + pad_l, src, n * sizeof(float));
    for (int i = 0; i < pad_r; ++i) dst[pad_l + n + i] = src[n - 2 - i];
}

}  // namespace

struct MelExtractor::Impl {
    MelConfig          cfg;
    std::vector<float> window;
    std::vector<float> mel_fb;   // [n_mels, n_fft/2+1]
};

MelExtractor::MelExtractor(const MelConfig & cfg) : impl_(std::make_unique<Impl>()) {
    if (cfg.win_length > cfg.n_fft) {
        throw InvalidArgument("MelExtractor: win_length must be <= n_fft");
    }
    impl_->cfg    = cfg;
    impl_->window = make_hann_window(cfg.win_length);
    impl_->mel_fb = make_mel_filterbank(cfg.n_fft, cfg.n_mels, cfg.sample_rate,
                                         cfg.fmin, cfg.fmax);
}

MelExtractor::~MelExtractor() = default;
MelExtractor::MelExtractor(MelExtractor &&) noexcept = default;
MelExtractor & MelExtractor::operator=(MelExtractor &&) noexcept = default;

const MelConfig & MelExtractor::config() const noexcept { return impl_->cfg; }

int MelExtractor::num_frames(std::size_t n_samples) const noexcept {
    const auto & cfg = impl_->cfg;
    const int pad_l = (cfg.win_length - cfg.hop_length) / 2;
    const int pad_r = (cfg.win_length - cfg.hop_length + 1) / 2;
    const std::int64_t padded = static_cast<std::int64_t>(n_samples) + pad_l + pad_r;
    const std::int64_t frames = (padded - cfg.win_length) / cfg.hop_length + 1;
    return frames > 0 ? static_cast<int>(frames) : 0;
}

std::vector<float> MelExtractor::forward(const float * wav, std::size_t n) const {
    const auto & cfg    = impl_->cfg;
    const auto & window = impl_->window;
    const auto & mel_fb = impl_->mel_fb;

    const int n_fft  = cfg.n_fft;
    const int win    = cfg.win_length;
    const int hop    = cfg.hop_length;
    const int n_mels = cfg.n_mels;
    const int n_bins = n_fft / 2 + 1;

    const int pad_l = (win - hop) / 2;
    const int pad_r = (win - hop + 1) / 2;
    std::vector<float> padded;
    reflect_pad(wav, n, pad_l, pad_r, padded);

    const int T = num_frames(n);
    if (T <= 0) return {};

    std::vector<float> out(static_cast<std::size_t>(T) * n_mels);

    // Frames are independent: split into stripes on a small worker pool
    // (≤ 8, guarded by frame count).  Each worker runs one batched pocketfft
    // r2c over its stripe to amortize twiddle-table setup across frames.
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    const int want  = static_cast<int>(std::min<unsigned>(hw, 8u));
    const int n_thr = std::max(1, std::min(want, T / 128 + 1));

    auto worker = [&](int t_begin, int t_end) {
        const int block = t_end - t_begin;
        std::vector<float> frame2d(static_cast<std::size_t>(n_fft) * block);
        std::vector<std::complex<float>> spec2d(
            static_cast<std::size_t>(n_bins) * block);

        // Windowing: fill the 2-D [n_fft, block] input (frames are column
        // contiguous; windowed region is [0, win), rest zero).
        for (int t = t_begin; t < t_end; ++t) {
            const std::size_t off = static_cast<std::size_t>(t) * hop;
            float * dst = frame2d.data() +
                static_cast<std::size_t>(t - t_begin) * n_fft;
            for (int k = 0; k < win; ++k) dst[k] = padded[off + k] * window[k];
            std::fill(dst + win, dst + n_fft, 0.0f);
        }

        pocketfft::shape_t  sh  = {static_cast<std::size_t>(n_fft),
                                   static_cast<std::size_t>(block)};
        // stride_t is vector<ptrdiff_t>: sizeof() yields size_t, which clang
        // rejects as a narrowing conversion in the initializer list.
        pocketfft::stride_t si  = {static_cast<std::ptrdiff_t>(sizeof(float)),
                                   static_cast<std::ptrdiff_t>(sizeof(float) * n_fft)};
        pocketfft::stride_t so  = {static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)),
                                   static_cast<std::ptrdiff_t>(sizeof(std::complex<float>) * n_bins)};
        pocketfft::shape_t  ax  = {0};
        pocketfft::r2c(sh, si, so, ax, pocketfft::FORWARD,
                       frame2d.data(), spec2d.data(), 1.0f);

        for (int t = t_begin; t < t_end; ++t) {
            const auto * sp = spec2d.data() +
                static_cast<std::size_t>(t - t_begin) * n_bins;
            float * dst = out.data() + static_cast<std::size_t>(t) * n_mels;
            for (int m = 0; m < n_mels; ++m) {
                const float * row = mel_fb.data() + static_cast<std::size_t>(m) * n_bins;
                float acc = 0.0f;
                for (int k = 0; k < n_bins; ++k) {
                    // Plain sqrt (not std::hypot): no overflow risk at FFT
                    // output magnitudes, and matches torch/librosa's |·|
                    // semantics within 1 ulp while running faster.
                    const float re = sp[k].real(), im = sp[k].imag();
                    acc += row[k] * std::sqrt(re * re + im * im);
                }
                dst[m] = std::log(std::max(acc, cfg.clip_val));
            }
        }
    };

    if (n_thr == 1) {
        worker(0, T);
        return out;
    }
    const int per = (T + n_thr - 1) / n_thr;
    std::vector<std::thread> pool;
    pool.reserve(n_thr);
    int t0 = 0;
    for (int i = 0; i < n_thr && t0 < T; ++i) {
        const int t1 = std::min(T, t0 + per);
        pool.emplace_back(worker, t0, t1);
        t0 = t1;
    }
    for (auto & th : pool) th.join();
    return out;
}

}  // namespace game_ggml
