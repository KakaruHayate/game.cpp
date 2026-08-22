#include "ops_joint_attn.h"

#include "ops_basic.h"
#include "ops_ffn.h"
#include "ops_rope.h"

#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace game_ggml::internal::ops {

// Convert FP32 → FP16 bit pattern using the standard "round to nearest even".
// Defined here so we don't pull a ggml-private header.
namespace {
std::uint16_t f32_to_f16_bits(float f) {
    // Equivalent to ggml_compute_fp32_to_fp16 used by ggml's mask kernels.
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    std::uint32_t sign = (x >> 16) & 0x8000;
    std::int32_t  exp  = static_cast<std::int32_t>((x >> 23) & 0xff) - 127 + 15;
    std::uint32_t mant =  (x & 0x007fffff);
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);  // underflow
        mant = (mant | 0x00800000) >> (1 - exp);
        if (mant & 0x00001000) mant += 0x00002000;   // round up
        return static_cast<std::uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 0x1f) {
        if (mant == 0) return static_cast<std::uint16_t>(sign | 0x7c00);  // inf
        return static_cast<std::uint16_t>(sign | 0x7c00 | (mant >> 13));  // nan
    }
    if (mant & 0x00001000) {
        mant += 0x00002000;
        if (mant & 0x00800000) { mant = 0; ++exp; }
    }
    return static_cast<std::uint16_t>(sign | (exp << 10) | (mant >> 13));
}
}  // namespace

std::vector<std::uint16_t> build_joint_attn_mask_fp16(
    const std::int32_t * regions, int T, int N)
{
    // For R=1 and B=1 (inference) we build an (N+T, N+T) bool mask:
    //   - pool[i] region id = i + 1 (1..N)
    //   - x[t] region id    = regions[t] (0 = padding, 1..N valid)
    //   - allowed iff (same_stream OR same_region(!=0)) AND both valid
    //
    // Layout: ggml flash_attn mask is (kv_seq, q_seq, 1, 1).  The (i=key, j=query)
    // element sits at index j*S + i where i is innermost (ne[0]=S).  So mask
    // "row" j is query j's allowed key set.
    //
    // Structure exploited for fast fill (regions is a monotone run-length
    // encoding — cumsum of boundaries — so same-region frames are contiguous):
    //   * pool query rows: keys [0, N) (same stream) + the x frames of the
    //     matching region (contiguous run) are zero; everything else -inf.
    //   * x query rows: pool key region-1 (single) + all valid x keys (same
    //     stream, zero-filled span-by-span) are zero; everything else -inf.
    // This replaces a per-element branchy double loop + per-element software
    // f16 conversion with bulk fills over contiguous runs.
    const int S = N + T;
    const std::uint16_t neg_inf_h = f32_to_f16_bits(-10000.0f);  // ggml convention for "blocked"
    const std::uint16_t zero_h    = f32_to_f16_bits(0.0f);

    std::vector<std::uint16_t> mask(static_cast<std::size_t>(S) * S, neg_inf_h);

    // Per-region [start, end) intervals of x frames (in x-key index space).
    std::vector<int> reg_start(static_cast<std::size_t>(N) + 2, -1);
    std::vector<int> reg_end(static_cast<std::size_t>(N) + 2, -1);
    for (int i = 0; i < T; ++i) {
        const int r = regions[i];
        if (r <= 0 || r > N) continue;             // padding / out-of-range
        if (reg_start[r] < 0) reg_start[r] = i;
        reg_end[r] = i + 1;
    }

    // Contiguous valid (non-padding) x spans — for the x-x same-stream block.
    // The same r in [1, N] predicate as the reg_start/reg_end pass above:
    // negative or > N ids are treated as invalid everywhere (a malformed id
    // must not be able to attend to valid x keys).
    struct Span { int b, e; };
    const auto is_valid_region = [N](int r) { return r >= 1 && r <= N; };
    std::vector<Span> x_spans;
    for (int i = 0; i < T; ) {
        if (!is_valid_region(regions[i])) { ++i; continue; }
        int e = i;
        while (e < T && is_valid_region(regions[e])) ++e;
        x_spans.push_back({i, e});
        i = e;
    }

    std::uint16_t * M = mask.data();

    // Pool query rows (j = 0..N-1): same-stream pool keys + matching region's x.
    for (int j = 0; j < N; ++j) {
        std::uint16_t * row = M + static_cast<std::size_t>(j) * S;
        std::fill(row, row + N, zero_h);           // pool-pool block (all valid)
        const int r = j + 1;
        if (reg_start[r] >= 0) {
            std::fill(row + N + reg_start[r], row + N + reg_end[r], zero_h);
        }
    }

    // X query rows (j = 0..T-1, key row N+j): single matching pool key +
    // same-stream valid x keys.  Padding/out-of-range query ids (incl.
    // negative or > N) admit nothing.
    for (int j = 0; j < T; ++j) {
        const int rj = regions[j];
        if (!is_valid_region(rj)) continue;      // padding query: nothing allowed
        std::uint16_t * row = M + static_cast<std::size_t>(N + j) * S;
        row[rj - 1] = zero_h;
        for (const Span & sp : x_spans) {
            std::fill(row + N + sp.b, row + N + sp.e, zero_h);
        }
    }
    return mask;
}

// ---------------------------------------------------------------------------
// Joint attention
// ---------------------------------------------------------------------------

namespace {

// Chunk a (3*D_total, n_tokens, 1) tensor into q, k, v each (D_total, n_tokens, 1).
struct QKVTriple { ggml_tensor * q; ggml_tensor * k; ggml_tensor * v; };

QKVTriple chunk_qkv_3(ggml_context * ctx, ggml_tensor * qkv) {
    const int64_t D_total = qkv->ne[0] / 3;
    const std::size_t esize = ggml_element_size(qkv);
    QKVTriple r;
    r.q = ggml_view_4d(ctx, qkv,
        D_total, qkv->ne[1], qkv->ne[2], qkv->ne[3],
        qkv->nb[1], qkv->nb[2], qkv->nb[3], 0);
    r.k = ggml_view_4d(ctx, qkv,
        D_total, qkv->ne[1], qkv->ne[2], qkv->ne[3],
        qkv->nb[1], qkv->nb[2], qkv->nb[3], D_total * esize);
    r.v = ggml_view_4d(ctx, qkv,
        D_total, qkv->ne[1], qkv->ne[2], qkv->ne[3],
        qkv->nb[1], qkv->nb[2], qkv->nb[3], 2 * D_total * esize);
    return r;
}

}  // namespace

JoinResult joint_attention(
    ggml_context * ctx,
    ggml_tensor * pool,                    // (D, N, 1)
    ggml_tensor * x,                       // (D, T, 1)
    const JointAttentionWeights & W,
    ggml_tensor * global_positions,        // int32 (N+T,)
    ggml_tensor * region_indices,          // int32 (N+T,)
    ggml_tensor * attn_mask_fp16,
    int num_heads,
    int head_dim,
    float theta,
    bool pool_only)
{
    const int64_t N = pool->ne[1];
    const int64_t T = x->ne[1];
    const int64_t S = N + T;
    const int64_t Dtot = num_heads * head_dim;

    // ---- project QKV on each stream ----
    ggml_tensor * pool_norm = rms_norm(ctx, pool, W.w_pool_norm);
    ggml_tensor * x_norm    = rms_norm(ctx, x,    W.w_x_norm);
    ggml_tensor * pool_qkv = linear(ctx, pool_norm, W.w_pool_qkv, W.b_pool_qkv);  // (3*Dtot, N, 1)
    ggml_tensor * x_qkv    = linear(ctx, x_norm,    W.w_x_qkv,    W.b_x_qkv);     // (3*Dtot, T, 1)

    auto pq_f = chunk_qkv_3(ctx, pool_qkv);
    auto xq_f = chunk_qkv_3(ctx, x_qkv);

    // Reshape to (head_dim, H, n_tokens, 1) so rope can operate (ne[2]=n_tokens).
    auto reshape_heads = [&](ggml_tensor * t, int64_t n_tok) {
        return ggml_reshape_4d(ctx, ggml_cont(ctx, t), head_dim, num_heads, n_tok, 1);
    };
    ggml_tensor * pool_q = reshape_heads(pq_f.q, N);
    ggml_tensor * pool_k = reshape_heads(pq_f.k, N);
    ggml_tensor * pool_v = reshape_heads(pq_f.v, N);
    ggml_tensor * x_q    = reshape_heads(xq_f.q, T);
    ggml_tensor * x_k    = reshape_heads(xq_f.k, T);
    ggml_tensor * x_v    = reshape_heads(xq_f.v, T);

    // qk_norm (per-head, along head_dim = ne[0]).
    pool_q = rms_norm(ctx, pool_q, W.w_pool_q_norm);
    pool_k = rms_norm(ctx, pool_k, W.w_pool_k_norm);
    x_q    = rms_norm(ctx, x_q,    W.w_x_q_norm);
    x_k    = rms_norm(ctx, x_k,    W.w_x_k_norm);

    // Concatenate pool and x along ne[2] (n_tokens).  Keys/values always span
    // the full S = N + T tokens; queries span only the pool rows when
    // pool_only (the x query rows would be discarded anyway).
    ggml_tensor * q = ggml_concat(ctx, pool_q, x_q, /*dim=*/2);     // (D_head, H, S, 1)
    ggml_tensor * k = ggml_concat(ctx, pool_k, x_k, /*dim=*/2);
    ggml_tensor * v = ggml_concat(ctx, pool_v, x_v, /*dim=*/2);
    if (pool_only) q = pool_q;                                     // (D_head, H, N, 1)

    // Mixed RoPE: lower-half head_dim rotated by global positions, upper-half
    // by region indices.  pool_only: q has only the N pool tokens, so the
    // rope positions must be truncated to the first N (pool rows first).
    if (pool_only) {
        ggml_tensor * gpos_n = ggml_view_1d(ctx, global_positions, N, 0);
        ggml_tensor * ridx_n = ggml_view_1d(ctx, region_indices, N, 0);
        q = region_rope(ctx, q, RegionRopeMode::Mixed,
                        gpos_n, ridx_n, nullptr, head_dim, theta);
    } else {
        q = region_rope(ctx, q, RegionRopeMode::Mixed,
                        global_positions, region_indices, nullptr, head_dim, theta);
    }
    k = region_rope(ctx, k, RegionRopeMode::Mixed,
                    global_positions, region_indices, nullptr, head_dim, theta);

    // Permute to (head_dim, tokens, H, 1) for flash_attn_ext.
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // pool_only: query rows are the first N rows of the full S×S mask
    // (ne[0]=k columns, ne[1]=q rows).  The strided view is not contiguous
    // in general (inherited nb2), so materialise a contiguous N-row copy.
    ggml_tensor * mask = attn_mask_fp16;
    if (pool_only) {
        ggml_tensor * mview = ggml_view_2d(ctx, attn_mask_fp16, S, N,
            attn_mask_fp16->nb[1], 0);
        mask = ggml_cont(ctx, mview);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask,
        scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    // out shape: (D_head, H, n_q, 1), n_q = S or N

    // Flatten heads to (H*D_head, n_q, 1).
    out = ggml_cont(ctx, out);
    out = ggml_reshape_3d(ctx, out, Dtot, pool_only ? N : S, 1);

    JoinResult r;
    if (pool_only) {
        // The whole output is the pool stream — no view/split needed.
        r.pool = linear(ctx, out, W.w_pool_out, W.b_pool_out);
        r.x    = nullptr;
    } else {
        // Split back into pool (first N tokens) and x (last T tokens).
        ggml_tensor * pool_chunk = ggml_view_3d(ctx, out,
            Dtot, N, 1, out->nb[1], out->nb[2], 0);
        ggml_tensor * x_chunk = ggml_view_3d(ctx, out,
            Dtot, T, 1, out->nb[1], out->nb[2], N * out->nb[1]);
        pool_chunk = ggml_cont(ctx, pool_chunk);
        x_chunk    = ggml_cont(ctx, x_chunk);
        r.pool = linear(ctx, pool_chunk, W.w_pool_out, W.b_pool_out);
        r.x    = linear(ctx, x_chunk,    W.w_x_out,    W.b_x_out);
    }
    return r;
}

// ---------------------------------------------------------------------------
// PJAC
// ---------------------------------------------------------------------------

namespace {
ggml_tensor * add_conv_bias_1d(ggml_context * ctx, ggml_tensor * y, ggml_tensor * b) {
    if (!b) return y;
    const std::size_t esize = ggml_element_size(b);
    const int64_t C = b->ne[0];
    ggml_tensor * b4 = ggml_view_4d(ctx, b, 1, C, 1, 1,
        esize, C * esize, C * esize, 0);
    return ggml_add(ctx, y, b4);
}
}

JoinResult pjac(
    ggml_context * ctx,
    ggml_tensor * pool,
    ggml_tensor * x,
    const PJACWeights & W,
    ggml_tensor * global_positions,
    ggml_tensor * region_indices,
    ggml_tensor * attn_mask_fp16,
    int num_heads,
    int head_dim,
    float theta,
    bool pool_only)
{
    // Attention branch — applied to pool / x directly (JointAttention has its
    // own pre-norm, so we don't pre-norm here).  pool_only restricts the
    // attention to the N pool query rows.
    JoinResult a = joint_attention(ctx, pool, x, W.jattn,
        global_positions, region_indices, attn_mask_fp16,
        num_heads, head_dim, theta, pool_only);

    // CgMLP branches — each has its own pre-norm.
    ggml_tensor * pool_cn = rms_norm(ctx, pool, W.w_c_norm_pool);
    ggml_tensor * c_pool  = cgmlp(ctx, pool_cn,
        W.w_cg_pool_pw1, W.b_cg_pool_pw1, W.w_cg_pool_norm,
        W.w_cg_pool_dw,  W.b_cg_pool_dw,
        W.w_cg_pool_pw2, W.b_cg_pool_pw2,
        W.cg_pool_kernel);

    // Merge per stream: [attn ; cgmlp] -> DW conv -> linear.
    auto merge_stream = [&](ggml_tensor * a_s, ggml_tensor * c_s,
                            ggml_tensor * w_merge, ggml_tensor * b_merge,
                            ggml_tensor * w_dw, ggml_tensor * b_dw, int kernel) {
        ggml_tensor * m = ggml_concat(ctx, a_s, c_s, /*dim=*/0);   // (2D, T, 1)
        if (kernel != 0 && w_dw) {
            ggml_tensor * m_tr = ggml_cont(ctx, ggml_permute(ctx, m, 1, 0, 2, 3));
            const int pad = (kernel - 1) / 2;
            ggml_tensor * cv = dwconv_1d(ctx, w_dw, m_tr, kernel, pad);
            cv = add_conv_bias_1d(ctx, cv, b_dw);
            ggml_tensor * cv_back = ggml_cont(ctx, ggml_permute(ctx, cv, 1, 0, 2, 3));
            m = ggml_add(ctx, cv_back, m);
        }
        return linear(ctx, m, w_merge, b_merge);
    };

    JoinResult r;
    r.pool = merge_stream(a.pool, c_pool,
        W.w_merge_pool, W.b_merge_pool,
        W.w_merge_dw_pool, W.b_merge_dw_pool, W.merge_pool_kernel);
    if (pool_only) {
        // x stream has no consumer: skip the x-side CgMLP + merge entirely.
        r.x = nullptr;
        return r;
    }
    ggml_tensor * x_cn = rms_norm(ctx, x, W.w_c_norm_x);
    ggml_tensor * c_x  = cgmlp(ctx, x_cn,
        W.w_cg_x_pw1, W.b_cg_x_pw1, W.w_cg_x_norm,
        W.w_cg_x_dw,  W.b_cg_x_dw,
        W.w_cg_x_pw2, W.b_cg_x_pw2,
        W.cg_x_kernel);
    r.x    = merge_stream(a.x, c_x,
        W.w_merge_x, W.b_merge_x,
        W.w_merge_dw_x, W.b_merge_dw_x, W.merge_x_kernel);
    return r;
}

// ---------------------------------------------------------------------------
// JEBF block
// ---------------------------------------------------------------------------

namespace {
ggml_tensor * apply_ffn_block(ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * w_norm,
    ggml_tensor * w_ln1, ggml_tensor * b_ln1,
    ggml_tensor * w_ln1_a, ggml_tensor * b_ln1_a, ggml_tensor * w_ln1_b, ggml_tensor * b_ln1_b,
    ggml_tensor * w_ln2, ggml_tensor * b_ln2,
    ggml_tensor * w_lay_scale)
{
    // lay_scale is folded into the ln2 linear at load time
    // (tensor_utils.cpp), so w_lay_scale is intentionally unused here.
    (void)w_lay_scale;
    ggml_tensor * h = rms_norm(ctx, x, w_norm);
    if (w_ln1_a) {
        // pre-split ln1 halves — two mul_mats, no cont copies.
        h = glu_ffn_split(ctx, h, w_ln1_a, b_ln1_a, w_ln1_b, b_ln1_b, w_ln2, b_ln2);
    } else {
        h = glu_ffn(ctx, h, w_ln1, b_ln1, w_ln2, b_ln2);
    }
    // JEBF uses `+ x` (not `* 0.5 + x`) unlike the single-stream EBF.
    return ggml_add(ctx, x, h);
}
}

JoinResult jebf_block(
    ggml_context * ctx,
    ggml_tensor * pool,
    ggml_tensor * x,
    const JEBFBlockWeights & W,
    ggml_tensor * global_positions,
    ggml_tensor * region_indices,
    ggml_tensor * attn_mask_fp16,
    int num_heads,
    int head_dim,
    float theta,
    bool pool_only)
{
    // --- FFN1 per stream ---
    if (W.has_ffn1) {
        // pool_only still needs FFN1_x: the x stream feeds the attention
        // keys (pool queries attend to the x keys), so x must carry the
        // FFN1-updated values.  Only FFN2_x (no consumer) is skipped below.
        x    = apply_ffn_block(ctx, x,    W.w_norm_ffn1_x,
            W.w_ffn1_x_ln1, W.b_ffn1_x_ln1,
            W.w_ffn1_x_ln1_a, W.b_ffn1_x_ln1_a, W.w_ffn1_x_ln1_b, W.b_ffn1_x_ln1_b,
            W.w_ffn1_x_ln2, W.b_ffn1_x_ln2,
            W.w_lay_scale_ffn1_x);
        pool = apply_ffn_block(ctx, pool, W.w_norm_ffn1_pool,
            W.w_ffn1_pool_ln1, W.b_ffn1_pool_ln1,
            W.w_ffn1_pool_ln1_a, W.b_ffn1_pool_ln1_a, W.w_ffn1_pool_ln1_b, W.b_ffn1_pool_ln1_b,
            W.w_ffn1_pool_ln2, W.b_ffn1_pool_ln2,
            W.w_lay_scale_ffn1_pool);
    }

    // --- PJAC ---
    // lay_scale_jpac_{x,pool} are folded into merge_linear_{x,pool} at load
    // time, so the residual uses att.* directly.  pool_only: attention runs on
    // the N pool query rows and the x stream is not produced.
    auto att = pjac(ctx, pool, x, W.pjac,
        global_positions, region_indices, attn_mask_fp16,
        num_heads, head_dim, theta, pool_only);
    if (!pool_only) x = ggml_add(ctx, x, att.x);
    pool = ggml_add(ctx, pool, att.pool);

    // --- FFN2 per stream ---
    if (W.has_ffn2) {
        if (!pool_only) {
            x    = apply_ffn_block(ctx, x,    W.w_norm_ffn2_x,
                W.w_ffn2_x_ln1, W.b_ffn2_x_ln1,
                W.w_ffn2_x_ln1_a, W.b_ffn2_x_ln1_a, W.w_ffn2_x_ln1_b, W.b_ffn2_x_ln1_b,
                W.w_ffn2_x_ln2, W.b_ffn2_x_ln2,
                W.w_lay_scale_ffn2_x);
        }
        pool = apply_ffn_block(ctx, pool, W.w_norm_ffn2_pool,
            W.w_ffn2_pool_ln1, W.b_ffn2_pool_ln1,
            W.w_ffn2_pool_ln1_a, W.b_ffn2_pool_ln1_a, W.w_ffn2_pool_ln1_b, W.b_ffn2_pool_ln1_b,
            W.w_ffn2_pool_ln2, W.b_ffn2_pool_ln2,
            W.w_lay_scale_ffn2_pool);
    }

    return {pool, x};
}

}  // namespace game_ggml::internal::ops
