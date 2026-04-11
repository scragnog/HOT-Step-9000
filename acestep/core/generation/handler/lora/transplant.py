"""On-the-fly adapter transplant between ACE-Step architectures (2B ↔ XL).

Detects when an adapter was trained on a different model architecture and
transparently transplants it at load time.  Supports both PEFT LoRA adapters
(factor-level A/B transplant) and LyCORIS LoKR adapters (dense-delta
reconstruction via Kronecker product + overlap-copy).

For LoRA: the overlapping region of lora_A / lora_B tensors is copied and
extra dimensions are zero-filled.  LoRA scaling correction is applied to
lora_B when rank/alpha ratios differ.

For LoKR: factor matrices are loaded directly from the safetensors file,
the dense weight delta is reconstructed via Kronecker product, and the
resulting delta is padded/cropped to match the target model's linear
dimensions — ready for direct use in merge-mode.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor, zeros_like


# ---------------------------------------------------------------------------
# LyCORIS key naming convention
# ---------------------------------------------------------------------------
# Keys: lycoris_<module_path_dots_to_underscores>.<suffix>
# Suffixes: .lokr_w1, .lokr_w2, .lokr_w1_a, .lokr_w1_b, .lokr_w2_a, .lokr_w2_b, .alpha
LYCORIS_PREFIX = "lycoris"

# PEFT LoRA key suffixes
_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"

# Module suffixes for LoKR factor tensors
_LOKR_SUFFIXES = (
    ".lokr_w1", ".lokr_w2",
    ".lokr_w1_a", ".lokr_w1_b",
    ".lokr_w2_a", ".lokr_w2_b",
    ".alpha",
)


@dataclass
class MismatchInfo:
    """Describes an architecture mismatch between adapter and model."""
    adapter_arch: str    # "2B" or "XL"
    model_arch: str      # "2B" or "XL"
    adapter_type: str    # "peft_lora" or "lycoris_lokr"
    mismatched_keys: int  # Number of keys with shape mismatches
    extra_layers: int     # Layers present in adapter but not model (or vice versa)
    detail: str           # Human-readable detail


# ---------------------------------------------------------------------------
# Overlap-copy (from user's transplant script)
# ---------------------------------------------------------------------------

def _copy_overlap_into_template(source: Tensor, template: Tensor) -> Tensor:
    """Copy the overlapping tensor slice from ``source`` into a zero template.

    Works for both padding (target bigger) and cropping (target smaller).
    """
    out = zeros_like(template)
    slices = tuple(
        slice(0, min(src_dim, dst_dim))
        for src_dim, dst_dim in zip(source.shape, template.shape)
    )
    out[slices] = source[slices].to(dtype=template.dtype)
    return out.contiguous()


# ---------------------------------------------------------------------------
# Key/module mapping helpers
# ---------------------------------------------------------------------------

def _lycoris_key_to_module_path(lycoris_key: str) -> str:
    """Convert a LyCORIS key prefix to a decoder module path.

    ``lycoris_layers_0_self_attn_q_proj`` → ``layers.0.self_attn.q_proj``
    ``lycoris_condition_embedder`` → ``condition_embedder``
    """
    # Strip the lycoris_ prefix
    name = lycoris_key
    if name.startswith(LYCORIS_PREFIX + "_"):
        name = name[len(LYCORIS_PREFIX) + 1:]

    # Reconstruct dotted path: underscores become dots, but numeric
    # segments after "layers" are layer indices (keep as-is).
    # Strategy: split on _, reconstruct with dots, but keep
    # multi-word module names together.
    #
    # Known module path segments (from ACE-Step architecture):
    #   layers.<N>.self_attn.q_proj
    #   layers.<N>.cross_attn.k_proj
    #   layers.<N>.mlp.gate_proj
    #   condition_embedder
    #   proj_in_1
    #   time_embed.linear_1
    #   time_embed.r_linear_1
    #   time_embed.time_proj
    #   time_embed.r_time_proj
    #
    # The underscore-to-dot conversion must reconstruct these.
    # LyCORIS uses: f"{LORA_PREFIX}_{name}".replace(".", "_")
    # So we need to reverse the dot→underscore replacement.
    # This is ambiguous in general, but we can use a heuristic:
    # insert dots before known submodule names.

    # Known submodule boundaries (order matters — longer prefixes first)
    _BOUNDARIES = [
        "self_attn", "cross_attn", "mlp",
        "time_embed", "proj_in",
        "condition_embedder",
    ]
    _SUBMODULE_NAMES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "linear_1", "linear_2",
        "r_linear_1", "r_linear_2",
        "time_proj", "r_time_proj",
    ]

    # First pass: restore dots around layer indices
    # lycoris key: layers_0_self_attn_q_proj → layers.0.self_attn.q_proj
    # Pattern: layers_<digits>_<rest>
    name = re.sub(r'layers_(\d+)_', r'layers.\1.', name)

    # Second pass: restore dots at known submodule boundaries
    for boundary in _BOUNDARIES:
        # "self_attn_q_proj" → "self_attn.q_proj"
        for sub in _SUBMODULE_NAMES:
            pattern = f"{boundary}_{sub}"
            replacement = f"{boundary}.{sub}"
            if pattern in name:
                name = name.replace(pattern, replacement)
                break

    # Third pass: time_embed boundaries
    # "time_embed_linear_1" → "time_embed.linear_1"
    name = re.sub(r'time_embed_(?=linear|time_proj|r_linear|r_time_proj)', 'time_embed.', name)

    # "proj_in_1" stays as "proj_in_1" (it's a single module name, not nested)

    return name


def _module_path_to_lycoris_key(module_path: str) -> str:
    """Convert a decoder module path to a LyCORIS key prefix.

    ``layers.0.self_attn.q_proj`` → ``lycoris_layers_0_self_attn_q_proj``
    """
    return f"{LYCORIS_PREFIX}_{module_path.replace('.', '_')}"


def _peft_key_to_module_path(peft_key: str) -> Optional[str]:
    """Extract the module path from a PEFT LoRA state dict key.

    ``base_model.model.layers.0.self_attn.q_proj.lora_A.weight``
    → ``layers.0.self_attn.q_proj``
    """
    for suffix in (_LORA_A_SUFFIX, _LORA_B_SUFFIX):
        if peft_key.endswith(suffix):
            path = peft_key[: -len(suffix)]
            return path.removeprefix("base_model.model.")
    return None


# ---------------------------------------------------------------------------
# Architecture mismatch detection
# ---------------------------------------------------------------------------

def _get_decoder_linear_shapes(decoder) -> Dict[str, Tuple[int, ...]]:
    """Return {module_path.weight: weight.shape} for all Linear layers in the decoder."""
    import torch.nn as nn
    shapes = {}
    for name, mod in decoder.named_modules():
        if isinstance(mod, nn.Linear):
            shapes[name + ".weight"] = tuple(mod.weight.shape)
    return shapes


def _infer_arch_label(num_layers: int, hidden_size: int) -> str:
    """Infer '2B' or 'XL' from layer count / hidden size."""
    if num_layers > 24 or hidden_size > 2048:
        return "XL"
    return "2B"


def detect_lokr_mismatch(
    decoder,
    weights_path: str,
) -> Optional[MismatchInfo]:
    """Detect if a LoKR safetensors file was trained on a different architecture.

    Compares:
    1. Number of transformer layers in adapter vs decoder
    2. Kronecker product output shapes vs decoder linear dimensions
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
    except Exception:
        return None

    # Parse module names from LoKR keys
    adapter_modules = set()
    for k in keys:
        base = re.sub(r'\.(lokr_w[12](_[ab])?|alpha)$', '', k)
        adapter_modules.add(base)

    # Get adapter layer count
    adapter_layers = set()
    for m in adapter_modules:
        match = re.search(r'layers_(\d+)', m)
        if match:
            adapter_layers.add(int(match.group(1)))

    # Get decoder layer count and shapes
    decoder_shapes = _get_decoder_linear_shapes(decoder)
    decoder_layers = set()
    for name in decoder_shapes:
        match = re.search(r'layers\.(\d+)', name)
        if match:
            decoder_layers.add(int(match.group(1)))

    # Quick check: if layer counts match and we can verify a few shapes, no mismatch
    layer_diff = len(adapter_layers) - len(decoder_layers)

    # Check shape mismatches on modules that exist in both
    mismatched = 0
    checked = 0
    for lycoris_mod in sorted(adapter_modules):
        module_path = _lycoris_key_to_module_path(lycoris_mod)
        decoder_key = module_path + ".weight"

        if decoder_key not in decoder_shapes:
            # Module doesn't exist in decoder — could be extra layers
            continue

        decoder_shape = decoder_shapes[decoder_key]

        # Compute expected LoKR output shape from factor matrices
        try:
            with safe_open(weights_path, framework="pt", device="cpu") as sf:
                w1_key = lycoris_mod + ".lokr_w1"
                w2_key = lycoris_mod + ".lokr_w2"
                w1a_key = lycoris_mod + ".lokr_w1_a"
                w2a_key = lycoris_mod + ".lokr_w2_a"
                w1b_key = lycoris_mod + ".lokr_w1_b"
                w2b_key = lycoris_mod + ".lokr_w2_b"

                sf_keys = sf.keys()

                if w1_key in sf_keys and w2_key in sf_keys:
                    w1 = sf.get_tensor(w1_key)
                    w2 = sf.get_tensor(w2_key)
                    kron_shape = (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1])
                elif w1_key in sf_keys and w2a_key in sf_keys and w2b_key in sf_keys:
                    w1 = sf.get_tensor(w1_key)
                    w2a = sf.get_tensor(w2a_key)
                    w2b = sf.get_tensor(w2b_key)
                    # w2 = w2a @ w2b → shape (w2a.shape[0], w2b.shape[1])
                    w2_shape = (w2a.shape[0], w2b.shape[1])
                    kron_shape = (w1.shape[0] * w2_shape[0], w1.shape[1] * w2_shape[1])
                elif w1a_key in sf_keys and w1b_key in sf_keys and w2_key in sf_keys:
                    w1a = sf.get_tensor(w1a_key)
                    w1b = sf.get_tensor(w1b_key)
                    w2 = sf.get_tensor(w2_key)
                    w1_shape = (w1a.shape[0], w1b.shape[1])
                    kron_shape = (w1_shape[0] * w2.shape[0], w1_shape[1] * w2.shape[1])
                elif w1a_key in sf_keys and w1b_key in sf_keys and w2a_key in sf_keys and w2b_key in sf_keys:
                    w1a = sf.get_tensor(w1a_key)
                    w1b = sf.get_tensor(w1b_key)
                    w2a = sf.get_tensor(w2a_key)
                    w2b = sf.get_tensor(w2b_key)
                    w1_shape = (w1a.shape[0], w1b.shape[1])
                    w2_shape = (w2a.shape[0], w2b.shape[1])
                    kron_shape = (w1_shape[0] * w2_shape[0], w1_shape[1] * w2_shape[1])
                else:
                    continue

                checked += 1
                if kron_shape != decoder_shape:
                    mismatched += 1

        except Exception:
            continue

    if mismatched == 0 and layer_diff == 0:
        return None

    # Determine architecture labels
    adapter_hidden = 0
    # Infer from adapter key dimensions — condition_embedder is (hidden_size, hidden_size)
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            cond_w1 = sf.get_tensor(f"{LYCORIS_PREFIX}_condition_embedder.lokr_w1")
            cond_w2 = sf.get_tensor(f"{LYCORIS_PREFIX}_condition_embedder.lokr_w2")
            adapter_hidden = cond_w1.shape[0] * cond_w2.shape[0]
    except Exception:
        pass

    # Get model hidden size from config or parameter shapes
    model_hidden = 0
    if "condition_embedder.weight" in decoder_shapes:
        model_hidden = decoder_shapes["condition_embedder.weight"][0]

    adapter_arch = _infer_arch_label(len(adapter_layers), adapter_hidden)
    model_arch = _infer_arch_label(len(decoder_layers), model_hidden)

    if adapter_arch == model_arch and mismatched == 0:
        return None  # Same architecture, no transplant needed

    extra = abs(layer_diff)
    direction = f"{adapter_arch} adapter → {model_arch} model"

    return MismatchInfo(
        adapter_arch=adapter_arch,
        model_arch=model_arch,
        adapter_type="lycoris_lokr",
        mismatched_keys=mismatched,
        extra_layers=extra,
        detail=f"{direction}: {mismatched}/{checked} checked modules have shape mismatches, "
               f"{extra} extra/missing layers",
    )


def detect_peft_lora_mismatch(
    decoder,
    adapter_path: str,
) -> Optional[MismatchInfo]:
    """Detect if a PEFT LoRA adapter was trained on a different architecture."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    model_path = os.path.join(adapter_path, "adapter_model.safetensors")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None

    try:
        from safetensors import safe_open

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        with safe_open(model_path, framework="pt", device="cpu") as sf:
            keys = list(sf.keys())

            decoder_shapes = _get_decoder_linear_shapes(decoder)
            mismatched = 0
            checked = 0

            for key in keys:
                module_path = _peft_key_to_module_path(key)
                if module_path is None:
                    continue

                decoder_key = module_path + ".weight"
                if decoder_key not in decoder_shapes:
                    continue

                tensor = sf.get_tensor(key)
                decoder_shape = decoder_shapes[decoder_key]
                checked += 1

                if key.endswith(_LORA_A_SUFFIX):
                    # lora_A: (rank, in_features) — in_features should match
                    if tensor.shape[1] != decoder_shape[1]:
                        mismatched += 1
                elif key.endswith(_LORA_B_SUFFIX):
                    # lora_B: (out_features, rank) — out_features should match
                    if tensor.shape[0] != decoder_shape[0]:
                        mismatched += 1

    except Exception:
        return None

    if mismatched == 0:
        return None

    # Infer adapter arch from first mismatched lora_A dimension
    adapter_hidden = 0
    model_hidden = 0
    try:
        with safe_open(model_path, framework="pt", device="cpu") as sf:
            for key in keys:
                if key.endswith(_LORA_A_SUFFIX):
                    t = sf.get_tensor(key)
                    adapter_hidden = max(adapter_hidden, t.shape[1])
        for shape in decoder_shapes.values():
            model_hidden = max(model_hidden, shape[0])
    except Exception:
        pass

    adapter_arch = _infer_arch_label(0, adapter_hidden)
    model_arch = _infer_arch_label(0, model_hidden)

    return MismatchInfo(
        adapter_arch=adapter_arch,
        model_arch=model_arch,
        adapter_type="peft_lora",
        mismatched_keys=mismatched,
        extra_layers=0,
        detail=f"{adapter_arch} adapter → {model_arch} model: {mismatched}/{checked} LoRA "
               f"tensors have dimension mismatches",
    )


# ---------------------------------------------------------------------------
# LoKR dense-delta transplant
# ---------------------------------------------------------------------------

def _compute_lokr_dense_delta_from_safetensors(
    weights_path: str,
) -> Dict[str, Tensor]:
    """Reconstruct dense weight deltas from LoKR factor matrices.

    Reads the safetensors file directly and computes the Kronecker product
    for each module — no LyCORIS model wrapping needed.

    Returns:
        Dict mapping decoder module paths (e.g. ``layers.0.self_attn.q_proj.weight``)
        to dense delta tensors.
    """
    from safetensors import safe_open

    with safe_open(weights_path, framework="pt", device="cpu") as sf:
        all_keys = set(sf.keys())

        # Group keys by module
        modules: Dict[str, Dict[str, Tensor]] = {}
        for k in sorted(all_keys):
            # Split on first dot after the module name prefix
            parts = k.rsplit(".", 1)
            if len(parts) != 2:
                continue
            mod_prefix, suffix = parts
            if mod_prefix not in modules:
                modules[mod_prefix] = {}
            modules[mod_prefix][suffix] = sf.get_tensor(k)

    deltas: Dict[str, Tensor] = {}

    for mod_prefix, tensors in modules.items():
        # Get alpha for scaling
        alpha_t = tensors.get("alpha")
        alpha = float(alpha_t.item()) if alpha_t is not None else 1.0

        # Reconstruct w1
        if "lokr_w1" in tensors:
            w1 = tensors["lokr_w1"].float()
        elif "lokr_w1_a" in tensors and "lokr_w1_b" in tensors:
            w1 = tensors["lokr_w1_a"].float() @ tensors["lokr_w1_b"].float()
        else:
            logger.warning(f"LoKR module {mod_prefix}: missing w1 factors, skipping")
            continue

        # Reconstruct w2
        if "lokr_w2" in tensors:
            w2 = tensors["lokr_w2"].float()
        elif "lokr_w2_a" in tensors and "lokr_w2_b" in tensors:
            w2 = tensors["lokr_w2_a"].float() @ tensors["lokr_w2_b"].float()
        else:
            logger.warning(f"LoKR module {mod_prefix}: missing w2 factors, skipping")
            continue

        # Compute dense delta via Kronecker product
        dense_delta = torch.kron(w1, w2)

        # Apply LyCORIS-compatible scaling.
        #
        # LyCORIS LoKR scaling depends on whether the factor matrices are
        # full (non-decomposed) or rank-decomposed:
        #
        # Case 1: Both w1 and w2 are full matrices (lokr_w1 + lokr_w2 keys)
        #   → LyCORIS forces alpha = lora_dim, making scale = 1.0
        #   → The stored alpha tensor is metadata-only, NOT used at inference
        #
        # Case 2: Either w1 or w2 is rank-decomposed (lokr_w1_a/b or w2_a/b)
        #   → scale = alpha / lora_dim, where lora_dim is the inner rank
        #     dimension of the decomposed factor
        #
        # This matches LyCORIS's LokrModule.__init__ logic:
        #   if self.use_w1 and self.use_w2: alpha = lora_dim  # force scale=1
        has_full_w1 = "lokr_w1" in tensors
        has_full_w2 = "lokr_w2" in tensors

        if has_full_w1 and has_full_w2:
            # Both full matrices → scale = 1.0 (LyCORIS convention)
            pass  # no scaling needed
        else:
            # At least one factor is rank-decomposed — extract lora_dim
            # from the inner dimension of the decomposed factor
            lora_dim = None
            if "lokr_w2_a" in tensors:
                # w2_a shape: (out_factor, lora_dim) or (lora_dim, out_factor)
                lora_dim = tensors["lokr_w2_a"].shape[1]
            elif "lokr_w2_b" in tensors:
                lora_dim = tensors["lokr_w2_b"].shape[0]
            elif "lokr_w1_a" in tensors:
                lora_dim = tensors["lokr_w1_a"].shape[1]
            elif "lokr_w1_b" in tensors:
                lora_dim = tensors["lokr_w1_b"].shape[0]

            if lora_dim is not None and lora_dim > 0:
                scale = alpha / lora_dim
                dense_delta = dense_delta * scale

        # Map to decoder module path
        module_path = _lycoris_key_to_module_path(mod_prefix)
        decoder_key = module_path + ".weight"

        if dense_delta.abs().max().item() > 1e-8:
            deltas[decoder_key] = dense_delta.detach()

    return deltas


def transplant_lokr_to_dense_delta(
    decoder,
    weights_path: str,
) -> Dict[str, Tensor]:
    """Transplant a LoKR adapter to dense deltas matching the decoder's architecture.

    1. Reconstructs dense deltas from LoKR factor matrices (Kronecker product)
    2. Pads/crops each delta to match the decoder's linear layer dimensions
    3. Returns deltas ready for use in merge-mode (_apply_merged_weights)
    """
    # Step 1: Compute source dense deltas directly from factor matrices
    source_deltas = _compute_lokr_dense_delta_from_safetensors(weights_path)

    if not source_deltas:
        logger.warning(f"No deltas reconstructed from {weights_path}")
        return {}

    # Step 2: Get decoder parameter shapes
    decoder_shapes = _get_decoder_linear_shapes(decoder)

    # Step 3: Transplant each delta
    transplanted: Dict[str, Tensor] = {}
    stats = {"exact": 0, "transplanted": 0, "skipped_no_match": 0}

    for key, source_delta in source_deltas.items():
        if key not in decoder_shapes:
            stats["skipped_no_match"] += 1
            logger.debug(f"  Transplant skip: {key} not in decoder")
            continue

        target_shape = decoder_shapes[key]

        if tuple(source_delta.shape) == target_shape:
            transplanted[key] = source_delta
            stats["exact"] += 1
        else:
            # Overlap-copy: pad or crop
            template = torch.zeros(target_shape, dtype=source_delta.dtype)
            transplanted[key] = _copy_overlap_into_template(source_delta, template)
            stats["transplanted"] += 1
            logger.debug(
                f"  Transplant: {key} {tuple(source_delta.shape)} → {target_shape}"
            )

    logger.info(
        f"LoKR transplant complete: {stats['exact']} exact, "
        f"{stats['transplanted']} padded/cropped, "
        f"{stats['skipped_no_match']} skipped (module not in decoder)"
    )

    return transplanted


# ---------------------------------------------------------------------------
# PEFT LoRA transplant
# ---------------------------------------------------------------------------

def _lora_scaling(config: Dict[str, Any], module_name: str, rank: int) -> float:
    """Return the PEFT LoRA scaling factor for one module."""
    alpha_pattern = config.get("alpha_pattern") or {}
    rank_pattern = config.get("rank_pattern") or {}
    alpha = alpha_pattern.get(module_name, config.get("lora_alpha", rank))
    module_rank = rank_pattern.get(module_name, rank)
    if config.get("use_rslora", False):
        return float(alpha) / float(module_rank) ** 0.5
    return float(alpha) / float(module_rank)


def transplant_peft_lora(
    decoder,
    adapter_path: str,
) -> Optional[str]:
    """Transplant a PEFT LoRA adapter to match the decoder's architecture.

    Returns the path to a temporary directory containing the transplanted
    adapter (adapter_config.json + adapter_model.safetensors), or None
    if transplant fails.  Caller must clean up the temp directory.
    """
    from safetensors.torch import load_file, save_file

    config_path = os.path.join(adapter_path, "adapter_config.json")
    model_path = os.path.join(adapter_path, "adapter_model.safetensors")

    with open(config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    source_state = load_file(model_path)
    decoder_shapes = _get_decoder_linear_shapes(decoder)

    # Build target template state dict
    transplanted: Dict[str, Tensor] = {}
    stats = {"exact": 0, "transplanted": 0, "scaled_lora_b": 0, "skipped": 0}

    for key, source_tensor in source_state.items():
        module_path = _peft_key_to_module_path(key)
        if module_path is None:
            # Non-LoRA tensor (e.g. bias), pass through
            transplanted[key] = source_tensor
            continue

        decoder_key = module_path + ".weight"
        if decoder_key not in decoder_shapes:
            transplanted[key] = source_tensor
            stats["skipped"] += 1
            continue

        decoder_shape = decoder_shapes[decoder_key]
        rank = adapter_config.get("r", source_tensor.shape[0])

        if key.endswith(_LORA_A_SUFFIX):
            # lora_A: (rank, in_features)
            target_shape = (rank, decoder_shape[1])
        elif key.endswith(_LORA_B_SUFFIX):
            # lora_B: (out_features, rank)
            target_shape = (decoder_shape[0], rank)
        else:
            transplanted[key] = source_tensor
            continue

        if tuple(source_tensor.shape) == target_shape:
            transplanted[key] = source_tensor
            stats["exact"] += 1
        else:
            template = torch.zeros(target_shape, dtype=source_tensor.dtype)
            transplanted[key] = _copy_overlap_into_template(source_tensor, template)
            stats["transplanted"] += 1

        # Apply LoRA scaling correction for lora_B when source and target
        # would produce different scaling factors
        if key.endswith(_LORA_B_SUFFIX):
            source_scale = _lora_scaling(adapter_config, module_path, rank)
            target_scale = _lora_scaling(adapter_config, module_path, rank)
            # Note: when using the same config for both, scales are equal.
            # This becomes relevant only if we adjust rank for the target.
            if abs(source_scale - target_scale) > 1e-8 and target_scale != 0:
                transplanted[key] = (
                    transplanted[key] * (source_scale / target_scale)
                ).contiguous()
                stats["scaled_lora_b"] += 1

    # Write to temp directory
    tmp_dir = tempfile.mkdtemp(prefix="transplant_lora_")
    try:
        # Copy adapter config (keep original settings)
        target_config = dict(adapter_config)
        target_config["inference_mode"] = True
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(target_config, f, indent=2, sort_keys=True)

        # Save transplanted weights with provenance metadata
        from safetensors import safe_open

        metadata = {}
        try:
            with safe_open(model_path, framework="pt", device="cpu") as handle:
                raw_meta = handle.metadata()
                metadata = dict(raw_meta) if raw_meta else {}
        except Exception:
            pass
        metadata["transplant_source"] = str(adapter_path)
        metadata["transplant_type"] = "peft_lora_architecture_transplant"

        save_file(transplanted, os.path.join(tmp_dir, "adapter_model.safetensors"), metadata=metadata)

        logger.info(
            f"PEFT LoRA transplant complete → {tmp_dir}: "
            f"{stats['exact']} exact, {stats['transplanted']} transplanted, "
            f"{stats['scaled_lora_b']} scaled"
        )
        return tmp_dir

    except Exception:
        # Clean up on failure
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# High-level API: detect + transplant
# ---------------------------------------------------------------------------

def detect_and_prepare_transplant(
    decoder,
    adapter_path: str,
    is_lokr: bool = False,
    weights_path: Optional[str] = None,
) -> Tuple[Optional[MismatchInfo], Optional[Dict[str, Tensor]], Optional[str]]:
    """Detect architecture mismatch and prepare transplant if needed.

    Args:
        decoder: The model's decoder module.
        adapter_path: Path to adapter directory (PEFT) or safetensors file (LoKR).
        is_lokr: Whether this is a LoKR adapter.
        weights_path: Explicit safetensors path for LoKR (if different from adapter_path).

    Returns:
        Tuple of:
        - MismatchInfo or None (if no mismatch)
        - Dense deltas dict or None (for LoKR transplant, merge-mode ready)
        - Temp dir path or None (for PEFT LoRA transplant, must be cleaned up)
    """
    if is_lokr:
        lokr_path = weights_path or adapter_path
        mismatch = detect_lokr_mismatch(decoder, lokr_path)
        if mismatch is None:
            return None, None, None

        logger.warning(
            f"⚠️ Architecture mismatch detected: {mismatch.detail}. "
            f"Auto-transplanting LoKR adapter — quality may differ from "
            f"natively-trained {mismatch.model_arch} adapters."
        )

        try:
            deltas = transplant_lokr_to_dense_delta(decoder, lokr_path)
            return mismatch, deltas, None
        except Exception as exc:
            logger.error(f"LoKR transplant failed: {exc}")
            return mismatch, None, None
    else:
        # PEFT LoRA
        mismatch = detect_peft_lora_mismatch(decoder, adapter_path)
        if mismatch is None:
            return None, None, None

        logger.warning(
            f"⚠️ Architecture mismatch detected: {mismatch.detail}. "
            f"Auto-transplanting LoRA adapter — quality may differ from "
            f"natively-trained {mismatch.model_arch} adapters."
        )

        try:
            tmp_dir = transplant_peft_lora(decoder, adapter_path)
            return mismatch, None, tmp_dir
        except Exception as exc:
            logger.error(f"PEFT LoRA transplant failed: {exc}")
            return mismatch, None, None
