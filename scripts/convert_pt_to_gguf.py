#!/usr/bin/env python3
"""Convert a GAME PyTorch checkpoint directory into a single GGUF file.

Usage:
    python convert_pt_to_gguf.py --model-dir GAME-pt-1.0-small -o game_small.gguf

Inputs:
    <model-dir>/model.pt         required; checkpoint produced by reduce.py
    <model-dir>/config.yaml      required; model + inference config
    <model-dir>/lang_map.json    required when model.use_languages is True

Output:
    a single FP32 GGUF file containing:
      * architecture metadata         (general.architecture = "game-me", etc.)
      * model hyperparameters         (game.model.*, game.encoder.*, ...)
      * inference / feature metadata  (game.inference.*)
      * language map as a JSON string (game.inference.lang_map)
      * every parameter tensor, flat-keyed (e.g. encoder.layers.0.attn.c.pw1.weight)

The converter is deliberately single-purpose: it does not quantize, does not
merge EMA (the shipped small checkpoint already has weights merged), and does
not depend on the rest of the GAME training environment.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml

try:
    import gguf  # type: ignore
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        "error: the 'gguf' Python package is required. Install with "
        "`pip install -r ggml_backend/scripts/requirements.txt`.\n"
    )
    raise

log = logging.getLogger("convert_pt_to_gguf")


# -----------------------------------------------------------------------------
# Expected architecture identity for this converter.
# Bump GAME_ARCH_VERSION when the on-disk GGUF schema changes in a
# backwards-incompatible way (new required metadata, different tensor layout).
# -----------------------------------------------------------------------------
GAME_ARCH = "game-me"
GAME_ARCH_VERSION = 2   # v2: attn q_linear/kv_linear fused into qkv (llama-style)


@dataclass
class ConversionReport:
    """Human-facing summary of the conversion."""

    num_tensors: int
    total_params: int
    metadata_keys: list[str]
    dropped_keys: list[str]
    stripped_prefix: str


# =============================================================================
# Helpers
# =============================================================================

def _load_state_dict(ckpt_path: pathlib.Path) -> dict[str, torch.Tensor]:
    """Load the (already-reduced) checkpoint's state dict.

    Mirrors the logic used by ``inference/api.load_state_dict_for_inference``:
    prefer the EMA shadow weights if present, but fall back cleanly when the
    checkpoint only contains ``state_dict``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict: dict[str, torch.Tensor] = dict(ckpt.get("state_dict", {}))
    ema: dict[str, torch.Tensor] = ckpt.get("ema_state_dict") or {}
    if ema:
        log.info("merging %d EMA tensors into state_dict", len(ema))
        state_dict.update(ema)
    if not state_dict:
        raise ValueError(f"checkpoint at {ckpt_path} has neither state_dict nor ema_state_dict")
    return state_dict


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    """Detect and strip a common ``model.`` prefix shared by every key.

    This avoids hard-coding the Lightning wrapper name into the GGUF schema.
    """
    if not state_dict:
        return state_dict, ""
    # Find the longest common prefix up to the first dot.
    first = next(iter(state_dict))
    prefix = first.split(".", 1)[0] + "." if "." in first else ""
    if prefix and all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}, prefix
    return state_dict, ""


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """Return a contiguous FP32 numpy array from a torch tensor."""
    arr = t.detach().cpu().contiguous().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    # Ensure contiguous (gguf writer assumes this).
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


# =============================================================================
# Expected-key audit: catch surprises before writing the GGUF.
# =============================================================================

def _expected_keys(cfg: dict[str, Any]) -> set[str]:
    """Enumerate the keys we expect a small GAME model (current config) to have.

    Called only as a sanity check; unknown keys are emitted with a warning but
    still written to the GGUF so the C++ loader can complain with the right
    key path in its error messages.
    """
    expected: set[str] = set()

    # Top-level
    expected.update({
        "spectrogram_projection.weight",
        "spectrogram_projection.bias",
        "noise_embedding.embedding.weight",
        "region_embedding.embedding.weight",
    })
    if cfg["model"].get("use_languages", True):
        expected.add("language_embedding.weight")
    if cfg["model"]["mode"] == "d3pm":
        expected.update({
            "time_embedding.0.weight",
            "time_embedding.0.bias",
            "time_embedding.2.weight",
            "time_embedding.2.bias",
        })

    # Encoder (EBFBackbone, 4 layers)
    expected.update(_ebf_backbone_keys("encoder", cfg["model"]["encoder"], return_latent=False))

    # Segmenter (EBFBackbone, N layers, return_latent=True)
    expected.update(_ebf_backbone_keys("segmenter", cfg["model"]["segmenter"], return_latent=True))

    # Estimator (JEBFBackbone)
    expected.update(_jebf_backbone_keys("estimator", cfg["model"]["estimator"]))

    return expected


def _ebf_backbone_keys(prefix: str, backbone_cfg: dict, *, return_latent: bool) -> set[str]:
    keys: set[str] = set()
    kwargs = backbone_cfg["kwargs"]
    num_layers = kwargs["num_layers"]
    skip_first_ffn = kwargs.get("skip_first_ffn", False)
    skip_out_ffn = kwargs.get("skip_out_ffn", False)
    use_ls = kwargs.get("use_ls", True)
    use_out_norm = kwargs.get("use_out_norm", True)

    keys.add(f"{prefix}.input_proj.weight")
    keys.add(f"{prefix}.input_proj.bias")
    for i in range(num_layers):
        block = f"{prefix}.layers.{i}"
        if not skip_first_ffn:
            keys.update({
                f"{block}.ffn1.ln1.weight", f"{block}.ffn1.ln1.bias",
                f"{block}.ffn1.ln2.weight", f"{block}.ffn1.ln2.bias",
                f"{block}.norm1.weight",
            })
        if not skip_out_ffn:
            keys.update({
                f"{block}.ffn2.ln1.weight", f"{block}.ffn2.ln1.bias",
                f"{block}.ffn2.ln2.weight", f"{block}.ffn2.ln2.bias",
                f"{block}.norm2.weight",
            })
        # PAC (self-attn + CgMLP merged).  q_linear/kv_linear are fused at
        # conversion time into a single qkv (llama-style): one [3*H*D, D]
        # linear instead of [H*D, D] + [2*H*D, D].
        keys.update({
            f"{block}.attn.attn.qkv.weight", f"{block}.attn.attn.qkv.bias",
            f"{block}.attn.attn.out_linear.weight", f"{block}.attn.attn.out_linear.bias",
            f"{block}.attn.c.pw1.weight", f"{block}.attn.c.pw1.bias",
            f"{block}.attn.c.norm.weight",
            f"{block}.attn.c.dw.weight", f"{block}.attn.c.dw.bias",
            f"{block}.attn.c.pw2.weight", f"{block}.attn.c.pw2.bias",
            f"{block}.attn.a_norm.weight",
            f"{block}.attn.c_norm.weight",
            f"{block}.attn.merge_linear.weight", f"{block}.attn.merge_linear.bias",
        })
        if kwargs.get("m_kernel_size", 31) != 0:
            keys.update({
                f"{block}.attn.merge_dw_conv.weight",
                f"{block}.attn.merge_dw_conv.bias",
            })
        if use_ls:
            if not skip_first_ffn:
                keys.add(f"{block}.lay_scale1.scale")
            keys.add(f"{block}.lay_scale2.scale")
            if not skip_out_ffn:
                keys.add(f"{block}.lay_scale3.scale")
    if return_latent:
        keys.update({
            f"{prefix}.latent_norm.weight",
            f"{prefix}.latent_proj.weight",
            f"{prefix}.latent_proj.bias",
        })
    if use_out_norm:
        keys.add(f"{prefix}.output_norm.weight")
    keys.update({
        f"{prefix}.output_proj.weight",
        f"{prefix}.output_proj.bias",
    })
    return keys


def _jebf_backbone_keys(prefix: str, backbone_cfg: dict) -> set[str]:
    keys: set[str] = set()
    kwargs = backbone_cfg["kwargs"]
    num_layers = kwargs["num_layers"]
    use_ls = kwargs.get("use_ls", True)
    use_out_norm = kwargs.get("use_out_norm", True)
    qk_norm = kwargs.get("qk_norm", True)
    skip_first_ffn = kwargs.get("skip_first_ffn", False)
    skip_out_ffn = kwargs.get("skip_out_ffn", False)

    keys.update({
        f"{prefix}.input_proj.weight",
        f"{prefix}.input_proj.bias",
        f"{prefix}.pool_token_gen.emb",
    })

    for i in range(num_layers):
        block = f"{prefix}.layers.{i}"
        if not skip_first_ffn:
            keys.update({
                f"{block}.ffn1_x.ln1.weight", f"{block}.ffn1_x.ln1.bias",
                f"{block}.ffn1_x.ln2.weight", f"{block}.ffn1_x.ln2.bias",
                f"{block}.ffn1_pool.ln1.weight", f"{block}.ffn1_pool.ln1.bias",
                f"{block}.ffn1_pool.ln2.weight", f"{block}.ffn1_pool.ln2.bias",
                f"{block}.norm_ffn1_x.weight", f"{block}.norm_ffn1_pool.weight",
            })
        if not skip_out_ffn:
            keys.update({
                f"{block}.ffn2_x.ln1.weight", f"{block}.ffn2_x.ln1.bias",
                f"{block}.ffn2_x.ln2.weight", f"{block}.ffn2_x.ln2.bias",
                f"{block}.ffn2_pool.ln1.weight", f"{block}.ffn2_pool.ln1.bias",
                f"{block}.ffn2_pool.ln2.weight", f"{block}.ffn2_pool.ln2.bias",
                f"{block}.norm_ffn2_x.weight", f"{block}.norm_ffn2_pool.weight",
            })
        # Joint attention (double-stream QKV + norms + out)
        keys.update({
            f"{block}.attn.jattn.pool_qkv.weight", f"{block}.attn.jattn.pool_qkv.bias",
            f"{block}.attn.jattn.x_qkv.weight",    f"{block}.attn.jattn.x_qkv.bias",
            f"{block}.attn.jattn.pool_out.weight", f"{block}.attn.jattn.pool_out.bias",
            f"{block}.attn.jattn.x_out.weight",    f"{block}.attn.jattn.x_out.bias",
            f"{block}.attn.jattn.pool_norm.weight",
            f"{block}.attn.jattn.x_norm.weight",
        })
        if qk_norm:
            keys.update({
                f"{block}.attn.jattn.pool_q_norm.weight",
                f"{block}.attn.jattn.pool_k_norm.weight",
                f"{block}.attn.jattn.x_q_norm.weight",
                f"{block}.attn.jattn.x_k_norm.weight",
            })
        # CgMLP (pool + x) + merge
        for stream in ("x", "pool"):
            keys.update({
                f"{block}.attn.c_{stream}.pw1.weight", f"{block}.attn.c_{stream}.pw1.bias",
                f"{block}.attn.c_{stream}.norm.weight",
                f"{block}.attn.c_{stream}.dw.weight",  f"{block}.attn.c_{stream}.dw.bias",
                f"{block}.attn.c_{stream}.pw2.weight", f"{block}.attn.c_{stream}.pw2.bias",
            })
        keys.update({
            f"{block}.attn.c_norm_x.weight",
            f"{block}.attn.c_norm_pool.weight",
            f"{block}.attn.merge_linear_x.weight",    f"{block}.attn.merge_linear_x.bias",
            f"{block}.attn.merge_linear_pool.weight", f"{block}.attn.merge_linear_pool.bias",
        })
        if kwargs.get("m_kernel_size_x", 31) != 0:
            keys.update({
                f"{block}.attn.merge_dw_conv_x.weight",
                f"{block}.attn.merge_dw_conv_x.bias",
            })
        if kwargs.get("m_kernel_size_pool", 5) != 0:
            keys.update({
                f"{block}.attn.merge_dw_conv_pool.weight",
                f"{block}.attn.merge_dw_conv_pool.bias",
            })
        if use_ls:
            if not skip_first_ffn:
                keys.update({
                    f"{block}.lay_scale_ffn1_x.scale",
                    f"{block}.lay_scale_ffn1_pool.scale",
                })
            if not skip_out_ffn:
                keys.update({
                    f"{block}.lay_scale_ffn2_x.scale",
                    f"{block}.lay_scale_ffn2_pool.scale",
                })
            keys.update({
                f"{block}.lay_scale_jpac_x.scale",
                f"{block}.lay_scale_jpac_pool.scale",
            })

    if use_out_norm:
        keys.update({
            f"{prefix}.output_norm_x.weight",
            f"{prefix}.output_norm_pool.weight",
        })
    keys.update({
        f"{prefix}.output_proj_x.weight",    f"{prefix}.output_proj_x.bias",
        f"{prefix}.output_proj_pool.weight", f"{prefix}.output_proj_pool.bias",
    })
    return keys


# =============================================================================
# Metadata writer
# =============================================================================

def _add_metadata(writer: gguf.GGUFWriter, cfg: dict, lang_map: dict[str, int] | None) -> list[str]:
    keys: list[str] = []

    def put(k: str, v):
        keys.append(k)
        if isinstance(v, bool):
            writer.add_bool(k, v)
        elif isinstance(v, int):
            writer.add_int32(k, v)
        elif isinstance(v, float):
            writer.add_float32(k, v)
        elif isinstance(v, str):
            writer.add_string(k, v)
        else:
            raise TypeError(f"unsupported metadata value for {k}: {type(v).__name__}")

    put("general.architecture", GAME_ARCH)
    put("general.name", cfg.get("_name", "GAME"))
    put("general.version", f"{GAME_ARCH_VERSION}")

    # --- model hyperparameters ---
    m = cfg["model"]
    put("game.model.mode",              m["mode"])                  # "d3pm" | "completion"
    put("game.model.embedding_dim",     int(m["embedding_dim"]))
    put("game.model.in_dim",            int(m["in_dim"]))
    put("game.model.estimator_out_dim", int(m["estimator_out_dim"]))
    put("game.model.region_cycle_len",  int(m.get("region_cycle_len", 3)))
    put("game.model.use_languages",     bool(m.get("use_languages", True)))
    put("game.model.num_languages",     int(m.get("num_languages", 0)))

    for section in ("encoder", "segmenter", "estimator"):
        sub = m[section]
        put(f"game.{section}.cls", sub["cls"])
        for k, v in sub["kwargs"].items():
            # Flatten kwargs keys directly under game.<section>.*
            put(f"game.{section}.{k}", v if not isinstance(v, (int, float, bool, str)) else v)

    # --- inference / feature config ---
    infer = cfg["inference"]
    feats = infer["features"]
    put("game.inference.audio_sample_rate", int(feats["audio_sample_rate"]))
    put("game.inference.hop_size",          int(feats["hop_size"]))
    put("game.inference.fft_size",          int(feats["fft_size"]))
    put("game.inference.win_size",          int(feats["win_size"]))
    put("game.inference.timestep",          float(feats["hop_size"] / feats["audio_sample_rate"]))
    spec = feats["spectrogram"]
    put("game.inference.spectrogram.type",     spec.get("type", "mel"))
    put("game.inference.spectrogram.num_bins", int(spec["num_bins"]))
    put("game.inference.spectrogram.fmin",     float(spec["fmin"]))
    put("game.inference.spectrogram.fmax",     float(spec["fmax"]))
    put("game.inference.midi_min",             float(infer["midi_min"]))
    put("game.inference.midi_max",             float(infer["midi_max"]))
    put("game.inference.midi_num_bins",        int(infer["midi_num_bins"]))
    put("game.inference.midi_std",             float(infer["midi_std"]))

    if lang_map is not None:
        put("game.inference.lang_map", json.dumps(lang_map, sort_keys=True, ensure_ascii=False))

    return keys


# =============================================================================
# Quantization
# =============================================================================

# GGML quantized storage requires the row width (ne0) to be a multiple of the
# type's block size.  The depthwise conv kernels (k=31 / 5 / 7) violate this,
# so they can only be stored as F16/F32.  ggml v0.11's conv_1d_dw additionally
# requires an F16 kernel, so we force F16 for every depthwise conv weight.
# Per-type block size (elements per block):
#   Q4_0/Q8_0 -> 32   (ggml-quants-legacy)
#   Q4_K/Q5_K/Q6_K -> 256  (K-superblocks)
_DW_WEIGHT_PATTERNS = (
    "*.attn.c.dw.weight",
    "*.attn.c_x.dw.weight",
    "*.attn.c_pool.dw.weight",
    "*.attn.merge_dw_conv.weight",
    "*.attn.merge_dw_conv_x.weight",
    "*.attn.merge_dw_conv_pool.weight",
)

# Consumed by ggml_repeat (learnable pool tokens copied to N regions): the
# repeat output inherits the source type, and the downstream rms_norm on the
# pool stream only supports F32 — so this must stay F32.
_F32_FORCED_PATTERNS = (
    "estimator.pool_token_gen.emb",
)

_QUANT_TYPES = {
    "F32": gguf.GGMLQuantizationType.F32,
    "F16": gguf.GGMLQuantizationType.F16,
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    "Q4_0": gguf.GGMLQuantizationType.Q4_0,
    "Q4_K": gguf.GGMLQuantizationType.Q4_K,
    "Q5_K": gguf.GGMLQuantizationType.Q5_K,
    "Q6_K": gguf.GGMLQuantizationType.Q6_K,
}

# Per-type block size in elements.  Q4_0/Q8_0 use 32; the K-superblocks use
# 256.  A row whose ne0 is not a multiple of its block size cannot be stored
# quantized and must fall back to F16.
_BLOCK_SIZES = {
    gguf.GGMLQuantizationType.Q4_0: 32,
    gguf.GGMLQuantizationType.Q8_0: 32,
    gguf.GGMLQuantizationType.Q4_K: 256,
    gguf.GGMLQuantizationType.Q5_K: 256,
    gguf.GGMLQuantizationType.Q6_K: 256,
}


@dataclass
class QuantRule:
    """A pattern -> type rule.  Later rules override earlier matches."""

    pattern: str
    qtype: gguf.GGMLQuantizationType


def parse_quant_config(path: pathlib.Path) -> list[QuantRule]:
    """Parse a JSON quant config file: {"<glob>": "<TYPE>", ...}."""
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("quant config must be a JSON object of {pattern: type}")
    rules: list[QuantRule] = []
    for pat, tname in raw.items():
        if tname not in _QUANT_TYPES:
            raise ValueError(
                f"unknown quant type '{tname}' for pattern '{pat}'; "
                f"valid: {sorted(_QUANT_TYPES)}"
            )
        rules.append(QuantRule(pat, _QUANT_TYPES[tname]))
    return rules


def _role_f32_only(name: str) -> bool:
    """Tensors consumed by elementwise ggml ops against F32 activations.

    rms_norm weights (ggml_mul), layer-scales (ggml_mul) and biases
    (ggml_add) all require the same type as the F32 activation tensor, so
    they must be stored F32 regardless of any quant rule.
    """
    return name.endswith(".bias") or name.endswith(".scale") or "norm" in name


def resolve_tensor_type(name: str, arr: np.ndarray, rules: list[QuantRule]) -> gguf.GGMLQuantizationType:
    """Pick the GGUF storage type for one tensor given the rule list.

    Hard constraints (GGML compatibility) always win:
      1. depthwise conv weights -> F16 (im2col_f16 requires it; k % block != 0)
      2. norms / lay-scales / biases -> F32 (elementwise ops, see _role_f32_only)
      3. any tensor whose ne0 % block(qtype) != 0 -> F16  (quant storage impossible)
      4. otherwise the last matching rule decides (default F32).
    """
    if any(fnmatch.fnmatch(name, p) for p in _DW_WEIGHT_PATTERNS):
        return gguf.GGMLQuantizationType.F16
    if any(fnmatch.fnmatch(name, p) for p in _F32_FORCED_PATTERNS):
        return gguf.GGMLQuantizationType.F32
    if _role_f32_only(name):
        return gguf.GGMLQuantizationType.F32

    qtype = gguf.GGMLQuantizationType.F32
    for r in rules:
        if fnmatch.fnmatch(name, r.pattern):
            qtype = r.qtype

    if qtype not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        # Q4_0/Q8_0 need 32-element blocks; Q4_K/Q5_K/Q6_K need 256.
        block = _BLOCK_SIZES[qtype]
        if arr.shape[-1] % block != 0:
            log.warning("  %s: ne0=%d %% %d != 0, falling back to F16",
                        name, arr.shape[-1], block)
            return gguf.GGMLQuantizationType.F16
    return qtype


def maybe_quantize(arr: np.ndarray, qtype: gguf.GGMLQuantizationType) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    """Quantize `arr` (torch layout, C-contiguous) for GGUF storage.

    Returns (payload_bytes_array, effective_qtype).  F32/F16 pass through as
    float32/float16 arrays; block quant types use gguf.quantize which assumes
    the data is laid out with the block axis last — identical to torch layout
    for (out, in) weights and (c, 1, k) convs, which is what GGML expects
    (ne0 = fastest dim).
    """
    if qtype == gguf.GGMLQuantizationType.F32:
        return arr.astype(np.float32, copy=False), qtype
    if qtype == gguf.GGMLQuantizationType.F16:
        return arr.astype(np.float16, copy=False), qtype
    return gguf.quantize(arr, qtype), qtype


# =============================================================================
# Main convert routine
# =============================================================================

def convert(model_dir: pathlib.Path, output_path: pathlib.Path, *, strict: bool,
            quant_rules: list[QuantRule] | None = None) -> ConversionReport:
    model_pt     = model_dir / "model.pt"
    config_yaml  = model_dir / "config.yaml"
    lang_map_js  = model_dir / "lang_map.json"

    if not model_pt.is_file():
        raise FileNotFoundError(model_pt)
    if not config_yaml.is_file():
        raise FileNotFoundError(config_yaml)

    log.info("reading config from %s", config_yaml)
    cfg = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))

    lang_map: dict[str, int] | None = None
    if cfg["model"].get("use_languages", True):
        if not lang_map_js.is_file():
            raise FileNotFoundError(
                f"{lang_map_js} is required when model.use_languages is True"
            )
        lang_map = json.loads(lang_map_js.read_text(encoding="utf-8"))

    cfg.setdefault("_name", model_dir.name)

    log.info("loading checkpoint %s", model_pt)
    raw = _load_state_dict(model_pt)
    sd, prefix = _strip_model_prefix(raw)
    log.info("stripped prefix '%s' from %d tensors", prefix, len(sd))

    # --- audit ---
    # q_linear/kv_linear are fused into qkv at write time; reflect that in the
    # expected-vs-present comparison so the audit sees the emitted names.
    def _fused_key(name: str) -> str:
        if name.endswith("attn.attn.q_linear.weight"):
            return name.replace("q_linear.weight", "qkv.weight")
        if name.endswith("attn.attn.q_linear.bias"):
            return name.replace("q_linear.bias", "qkv.bias")
        if name.endswith("attn.attn.kv_linear.weight"):
            return name.replace("kv_linear.weight", "qkv.weight")
        if name.endswith("attn.attn.kv_linear.bias"):
            return name.replace("kv_linear.bias", "qkv.bias")
        return name

    expected = _expected_keys(cfg)
    present  = {_fused_key(k) for k in sd}
    missing  = sorted(expected - present)
    extras   = sorted(present - expected)
    if missing:
        msg = f"{len(missing)} expected tensors missing, e.g. {missing[:5]}"
        if strict:
            raise KeyError(msg)
        log.warning(msg)
    if extras:
        log.warning("%d unexpected tensors present, e.g. %s", len(extras), extras[:5])

    # --- write ---
    log.info("writing GGUF to %s", output_path)
    writer = gguf.GGUFWriter(str(output_path), GAME_ARCH)
    metadata_keys = _add_metadata(writer, cfg, lang_map)

    total_params = 0
    quant_report: dict[str, str] = {}

    # --- QKV fusion (llama-style): emit q_linear⊗kv_linear as one qkv tensor ---
    # Rows are quant-block aligned on the in-dim and identical whether packed
    # or separate, so Q8/etc. numbers are bit-identical to the un-fused form.
    # Sorted state-dict order visits kv_linear before q_linear; whichever of
    # the pair comes first fuses both and marks the sibling consumed.
    writes: list[tuple[str, np.ndarray]] = []
    consumed: set[str] = set()
    for name in sorted(sd):
        if name in consumed:
            continue
        if name.endswith("attn.attn.q_linear.weight"):
            kv = name.replace("q_linear.weight", "kv_linear.weight")
            consumed.add(kv)
            arr = np.concatenate([_tensor_to_np(sd[name]), _tensor_to_np(sd[kv])], axis=0)
            writes.append((name.replace("q_linear.weight", "qkv.weight"), arr))
        elif name.endswith("attn.attn.kv_linear.weight"):
            q = name.replace("kv_linear.weight", "q_linear.weight")
            consumed.add(q)
            arr = np.concatenate([_tensor_to_np(sd[q]), _tensor_to_np(sd[name])], axis=0)
            writes.append((name.replace("kv_linear.weight", "qkv.weight"), arr))
        elif name.endswith("attn.attn.q_linear.bias"):
            kv = name.replace("q_linear.bias", "kv_linear.bias")
            consumed.add(kv)
            arr = np.concatenate([_tensor_to_np(sd[name]), _tensor_to_np(sd[kv])], axis=0)
            writes.append((name.replace("q_linear.bias", "qkv.bias"), arr))
        elif name.endswith("attn.attn.kv_linear.bias"):
            q = name.replace("kv_linear.bias", "q_linear.bias")
            consumed.add(q)
            arr = np.concatenate([_tensor_to_np(sd[q]), _tensor_to_np(sd[name])], axis=0)
            writes.append((name.replace("kv_linear.bias", "qkv.bias"), arr))
        else:
            writes.append((name, _tensor_to_np(sd[name])))
    del consumed

    for name, arr in writes:
        qtype = resolve_tensor_type(name, arr, quant_rules or [])
        payload, eff = maybe_quantize(arr, qtype)
        # For block-quantized payloads the writer expects the *byte* shape and
        # derives the logical shape itself; F32/F16 pass-through uses the data
        # shape directly, so payload.shape is correct in both cases.
        writer.add_tensor(name, payload, raw_shape=payload.shape, raw_dtype=eff)
        total_params += int(arr.size)
        quant_report[name] = eff.name
    if quant_rules:
        n_q = sum(1 for v in quant_report.values() if v not in ("F32", "F16"))
        log.info("quantization: %d/%d tensors stored as block-quantized types",
                 n_q, len(quant_report))
        for t in sorted({v for v in quant_report.values()}):
            log.info("  %-6s %4d tensors", t, sum(1 for v in quant_report.values() if v == t))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    report = ConversionReport(
        num_tensors=len(sd),
        total_params=total_params,
        metadata_keys=metadata_keys,
        dropped_keys=extras if not strict else [],
        stripped_prefix=prefix,
    )
    log.info("wrote %d tensors, %d metadata entries, %s params",
             report.num_tensors, len(report.metadata_keys), f"{report.total_params:,}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=pathlib.Path, required=True,
                        help="Directory containing model.pt + config.yaml + (lang_map.json)")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True,
                        help="Path to the output GGUF file")
    parser.add_argument("--quant-config", type=pathlib.Path, default=None,
                        help="JSON file: {<tensor glob>: <TYPE>} with TYPE in "
                             "F32/F16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K. Hard GGML "
                             "constraints override: depthwise convs -> F16, "
                             "norms/scales/biases -> F32, ne0%%32!=0 -> F16.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any expected tensor is missing (default: warn)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        quant_rules = parse_quant_config(args.quant_config)
        report = convert(args.model_dir, args.output, strict=args.strict,
                         quant_rules=quant_rules)
    except Exception as e:
        log.error("conversion failed: %s", e)
        return 1

    mb = args.output.stat().st_size / (1024 * 1024)
    print(f"✓ wrote {args.output} ({mb:.1f} MB, "
          f"{report.num_tensors} tensors, {report.total_params:,} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
