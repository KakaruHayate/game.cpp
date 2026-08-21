#include "slicer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace game_ggml::cli {

namespace {

// Framewise RMS (centered padding — matches librosa defaults used by slicer2.py).
std::vector<float> frame_rms(const float * x, std::size_t n,
                             std::size_t frame_len, std::size_t hop_len) {
    const std::size_t pad_l = frame_len / 2;
    const std::size_t pad_r = frame_len / 2;
    std::vector<float> padded(n + pad_l + pad_r, 0.0f);
    for (std::size_t i = 0; i < n; ++i) padded[pad_l + i] = x[i];
    // Constant padding with zeros (librosa default "constant").

    const std::size_t n_frames = (padded.size() - frame_len) / hop_len + 1;
    std::vector<float> rms(n_frames);
    for (std::size_t f = 0; f < n_frames; ++f) {
        double acc = 0.0;
        const float * p = padded.data() + f * hop_len;
        for (std::size_t k = 0; k < frame_len; ++k) {
            const double v = static_cast<double>(p[k]);
            acc += v * v;
        }
        rms[f] = static_cast<float>(std::sqrt(acc / static_cast<double>(frame_len)));
    }
    return rms;
}

std::size_t argmin(const float * v, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) if (v[i] < v[best]) best = i;
    return best;
}

}  // namespace

std::vector<SliceChunk> slice_waveform(
    const float * samples, std::size_t n,
    const SlicerConfig & cfg)
{
    // Translate ms-based knobs into frames (all using hop_size as the unit).
    const int sr = cfg.sample_rate;
    const int hop_size = std::max(1, sr * cfg.hop_ms / 1000);

    // min_interval in frames of hop_size; win_size bounded by 4*hop_size.
    const double min_interval_samples = sr * cfg.min_interval_ms / 1000.0;
    const int win_size = std::min<int>(
        static_cast<int>(std::round(min_interval_samples)), 4 * hop_size);

    const int min_length   = static_cast<int>(
        std::round(static_cast<double>(sr) * cfg.min_length_ms / 1000.0 / hop_size));
    const int min_interval = static_cast<int>(
        std::round(min_interval_samples / hop_size));
    const int max_sil_kept = static_cast<int>(
        std::round(static_cast<double>(sr) * cfg.max_sil_kept_ms / 1000.0 / hop_size));
    const float threshold = std::pow(10.0f, cfg.threshold_db / 20.0f);

    // Emit everything as a single chunk when the clip is too short.
    if (static_cast<int>((n + hop_size - 1) / hop_size) <= min_length) {
        SliceChunk chunk;
        chunk.offset_seconds = 0.0;
        chunk.waveform.assign(samples, samples + n);
        return {chunk};
    }

    auto rms = frame_rms(samples, n, win_size, hop_size);
    const int n_frames = static_cast<int>(rms.size());

    // Collect silence-cut tags (matches slicer2.py step by step).
    std::vector<std::pair<int,int>> sil_tags;   // {begin, end} silent-frame index range
    int silence_start = -1;
    int clip_start    = 0;

    for (int i = 0; i < n_frames; ++i) {
        if (rms[i] < threshold) {
            if (silence_start < 0) silence_start = i;
            continue;
        }
        if (silence_start < 0) continue;

        const bool is_leading_silence   = (silence_start == 0 && i > max_sil_kept);
        const bool need_slice_middle    = (i - silence_start >= min_interval) &&
                                          (i - clip_start    >= min_length);
        if (!is_leading_silence && !need_slice_middle) {
            silence_start = -1;
            continue;
        }

        if (i - silence_start <= max_sil_kept) {
            const int pos = static_cast<int>(argmin(rms.data() + silence_start,
                                                    i - silence_start + 1)) + silence_start;
            if (silence_start == 0) {
                sil_tags.push_back({0, pos});
            } else {
                sil_tags.push_back({pos, pos});
            }
            clip_start = pos;
        } else if (i - silence_start <= max_sil_kept * 2) {
            // Range [i - max_sil_kept, silence_start + max_sil_kept + 1]
            const int lo = i - max_sil_kept;
            const int hi = std::min<int>(n_frames - 1, silence_start + max_sil_kept);
            const int pos = static_cast<int>(argmin(rms.data() + lo, hi - lo + 1)) + lo;
            const int pos_l = static_cast<int>(argmin(rms.data() + silence_start,
                max_sil_kept + 1)) + silence_start;
            const int pos_r = static_cast<int>(argmin(rms.data() + (i - max_sil_kept),
                max_sil_kept + 1)) + i - max_sil_kept;
            if (silence_start == 0) {
                sil_tags.push_back({0, pos_r});
                clip_start = pos_r;
            } else {
                sil_tags.push_back({std::min(pos_l, pos), std::max(pos_r, pos)});
                clip_start = std::max(pos_r, pos);
            }
        } else {
            const int pos_l = static_cast<int>(argmin(rms.data() + silence_start,
                max_sil_kept + 1)) + silence_start;
            const int pos_r = static_cast<int>(argmin(rms.data() + (i - max_sil_kept),
                max_sil_kept + 1)) + i - max_sil_kept;
            if (silence_start == 0) sil_tags.push_back({0, pos_r});
            else                    sil_tags.push_back({pos_l, pos_r});
            clip_start = pos_r;
        }
        silence_start = -1;
    }

    if (silence_start >= 0 && n_frames - silence_start >= min_interval) {
        // Bounded argmin window: silence_end must be a valid rms index
        // (0..n_frames-1).  The old `min(n_frames, ...)` let the argmin read
        // rms[n_frames] — one past the end, an uninitialised heap value whose
        // luck-of-the-draw minimum flipped the trailing-silence cut position
        // between runs (e.g. 452 vs 501 -> 9.04 s vs 10.0 s chunks).
        const int silence_end = std::min<int>(n_frames - 1, silence_start + max_sil_kept);
        const int pos = static_cast<int>(argmin(rms.data() + silence_start,
            silence_end - silence_start + 1)) + silence_start;
        sil_tags.push_back({pos, n_frames + 1});
    }

    // Build chunks from the silence tags.
    auto apply_slice = [&](int begin, int end) -> SliceChunk {
        const std::size_t b = static_cast<std::size_t>(begin) * hop_size;
        const std::size_t e = std::min<std::size_t>(n, static_cast<std::size_t>(end) * hop_size);
        SliceChunk c;
        c.offset_seconds = static_cast<double>(b) / sr;
        c.waveform.assign(samples + b, samples + e);
        return c;
    };

    std::vector<SliceChunk> chunks;
    if (sil_tags.empty()) {
        chunks.push_back({0.0, std::vector<float>(samples, samples + n)});
        return chunks;
    }
    if (sil_tags[0].first > 0) chunks.push_back(apply_slice(0, sil_tags[0].first));
    for (std::size_t i = 0; i + 1 < sil_tags.size(); ++i) {
        chunks.push_back(apply_slice(sil_tags[i].second, sil_tags[i + 1].first));
    }
    if (static_cast<std::size_t>(sil_tags.back().second) < static_cast<std::size_t>(n_frames)) {
        chunks.push_back(apply_slice(sil_tags.back().second, n_frames));
    }
    if (chunks.empty()) {
        chunks.push_back({0.0, std::vector<float>(samples, samples + n)});
    }
    return chunks;
}

}  // namespace game_ggml::cli
