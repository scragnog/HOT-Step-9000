"""LoRA/LoKr adapter load/unload lifecycle management."""

import gc
import json
import dataclasses
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log
from acestep.core.generation.handler.lora.lokr_config import LoKRConfig

LOKR_WEIGHTS_FILENAME = "lokr_weights.safetensors"


def _flush_vram(label: str = "") -> None:
    """Force Python GC and release cached CUDA/MPS memory back to the OS.

    Call at every VRAM-sensitive transition point (before/after PEFT wrapping,
    after state_dict backup, after adapter unload, etc.) to reclaim fragmented
    blocks and maximize contiguous free VRAM.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.debug(
                f"VRAM flush{f' ({label})' if label else ''}: "
                f"allocated={alloc:.2f}GB, reserved={reserved:.2f}GB"
            )
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _try_prepare_bare_peft_safetensors(safetensors_path: str) -> Optional[str]:
    """Detect a bare PEFT LoRA safetensors file and prepare a temp loading directory.

    Some LoRA adapters (e.g. ComfyUI exports) ship as a single ``.safetensors``
    file with PEFT-compatible weight keys (``lora_down.weight``, ``lora_up.weight``)
    but *no* ``adapter_config.json`` and *no* embedded metadata.

    This function:
    1. Opens the file and checks for PEFT-style key naming
    2. Infers rank, alpha, target_modules, and DoRA from the weight structure
    3. Creates a temporary directory with ``adapter_config.json`` + a hard-link
       (or copy) of the weights as ``adapter_model.safetensors``
    4. Returns the temp directory path (caller must clean up), or ``None`` if
       the file doesn't look like a PEFT LoRA.
    """
    if not os.path.isfile(safetensors_path) or not safetensors_path.lower().endswith(".safetensors"):
        return None

    try:
        from safetensors import safe_open
    except ImportError:
        return None

    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as sf:
            keys = list(sf.keys())

            # Quick check: does this look like PEFT LoRA?
            lora_down_keys = [k for k in keys if "lora_down.weight" in k]
            if not lora_down_keys:
                return None

            # --- Infer rank from first lora_down weight shape ---
            first_down = sf.get_tensor(lora_down_keys[0])
            rank = first_down.shape[0]

            # --- Infer alpha from scalar alpha tensors ---
            alpha_keys = [k for k in keys if k.endswith(".alpha")]
            lora_alpha = float(rank)  # default: alpha == rank
            if alpha_keys:
                alpha_val = sf.get_tensor(alpha_keys[0])
                lora_alpha = float(alpha_val.item())

            # --- Detect DoRA ---
            use_dora = any("dora_scale" in k for k in keys)

            # --- Infer target modules from key naming ---
            target_modules = set()
            for k in lora_down_keys:
                # Key pattern: base_model.model.layers.N.<module_path>.lora_down.weight
                parts = k.split(".")
                try:
                    idx = parts.index("lora_down")
                    target_modules.add(parts[idx - 1])
                except (ValueError, IndexError):
                    pass

            if not target_modules:
                return None

    except Exception as exc:
        logger.debug(f"Failed to inspect safetensors for PEFT detection: {exc}")
        return None

    # --- Build temp dir with adapter_config.json + converted weights ---
    # ComfyUI/Kohya format uses different key names than PEFT:
    #   lora_down.weight → lora_A.weight
    #   lora_up.weight   → lora_B.weight
    #   dora_scale       → lora_magnitude_vector
    #   .alpha (tensors) → removed (stored in adapter_config.json instead)
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file as st_save_file

        # Re-open and convert all tensors
        converted_tensors = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)
                # Skip alpha scalars — already captured in config
                if key.endswith(".alpha"):
                    continue
                # Rename ComfyUI keys → PEFT keys
                new_key = key
                new_key = new_key.replace(".lora_down.weight", ".lora_A.weight")
                new_key = new_key.replace(".lora_up.weight", ".lora_B.weight")
                new_key = new_key.replace(".dora_scale", ".lora_magnitude_vector")
                converted_tensors[new_key] = tensor

        tmp_dir = tempfile.mkdtemp(prefix="peft_adapter_")
        config = {
            "peft_type": "LORA",
            "auto_mapping": None,
            "base_model_name_or_path": "",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_to_transform": None,
            "layers_pattern": None,
            "lora_alpha": lora_alpha,
            "lora_dropout": 0,
            "modules_to_save": None,
            "r": rank,
            "revision": None,
            "target_modules": sorted(target_modules),
            "task_type": None,
            "use_dora": use_dora,
        }
        config_path = os.path.join(tmp_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        dest_weights = os.path.join(tmp_dir, "adapter_model.safetensors")
        st_save_file(converted_tensors, dest_weights)
        del converted_tensors

        logger.info(
            f"Inferred PEFT config from bare safetensors: r={rank}, alpha={lora_alpha}, "
            f"targets={sorted(target_modules)}, dora={use_dora} → {tmp_dir}"
        )
        return tmp_dir
    except Exception as exc:
        logger.warning(f"Failed to create temp PEFT adapter directory: {exc}")
        return None


def _is_lokr_safetensors(weights_path: str) -> bool:
    """Return whether ``weights_path`` looks like a LoKr/LyCORIS safetensors file."""
    if not os.path.isfile(weights_path) or not weights_path.lower().endswith(".safetensors"):
        return False
    if os.path.basename(weights_path) == LOKR_WEIGHTS_FILENAME:
        return True

    try:
        from safetensors import safe_open
    except ImportError:
        return False

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            metadata: dict[str, Any] = sf.metadata() or {}
    except Exception:
        return False

    raw_config = metadata.get("lokr_config")
    return isinstance(raw_config, str) and bool(raw_config.strip())


def _resolve_lokr_weights_path(adapter_path: str) -> str | None:
    """Return LoKr safetensors path when ``adapter_path`` points to LoKr artifacts."""
    if os.path.isfile(adapter_path):
        return adapter_path if _is_lokr_safetensors(adapter_path) else None
    if os.path.isdir(adapter_path):
        weights_path = os.path.join(adapter_path, LOKR_WEIGHTS_FILENAME)
        if os.path.exists(weights_path):
            return weights_path

        # Backward-compat: support custom LyCORIS safetensors filenames that
        # carry ``lokr_config`` metadata.
        try:
            entries = os.listdir(adapter_path)
        except OSError:
            return None
        for name in entries:
            candidate = os.path.join(adapter_path, name)
            if _is_lokr_safetensors(candidate):
                return candidate
    return None


def _read_trigger_word_from_safetensors(
    weights_path: str,
) -> Tuple[str, str]:
    """Extract trigger_word and tag_position from safetensors __metadata__.

    Returns:
        Tuple of (trigger_word, tag_position) — both empty strings if absent.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return ("", "")

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            metadata: dict[str, Any] = sf.metadata() or {}
    except Exception:
        return ("", "")

    raw_config = metadata.get("lokr_config", "")
    if not isinstance(raw_config, str) or not raw_config.strip():
        return ("", "")

    try:
        parsed = json.loads(raw_config)
    except (json.JSONDecodeError, TypeError):
        return ("", "")

    if not isinstance(parsed, dict):
        return ("", "")

    trigger_word = parsed.get("trigger_word", "")
    tag_position = parsed.get("tag_position", "")
    return (trigger_word or "", tag_position or "")


def _load_lokr_config(weights_path: str) -> LoKRConfig:
    """Build ``LoKRConfig`` from safetensors metadata, with defaults on parse failure."""
    config = LoKRConfig()
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors metadata reader unavailable; using default LoKr config.")
        return config

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            metadata: dict[str, Any] = sf.metadata() or {}
    except Exception as exc:
        logger.warning(f"Unable to read LoKr metadata from {weights_path}: {exc}")
        return config

    raw_config = metadata.get("lokr_config")
    if not isinstance(raw_config, str) or not raw_config.strip():
        return config

    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        logger.warning(f"Invalid LoKr metadata config JSON in {weights_path}: {exc}")
        return config

    if not isinstance(parsed, dict):
        return config

    allowed_keys = set(LoKRConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in parsed.items() if k in allowed_keys}
    if not filtered:
        return config

    try:
        return LoKRConfig(**filtered)
    except Exception as exc:
        logger.warning(f"Failed to apply LoKr metadata config from {weights_path}: {exc}")
        return config


def _load_lokr_adapter(decoder: Any, weights_path: str) -> Any:
    """Inject and load a LoKr LyCORIS adapter into ``decoder``.

    Uses ``create_lycoris_from_weights`` to reconstruct the exact network
    structure directly from the saved weight keys.  This avoids silent
    mismatches that occur when the metadata ``lokr_config`` is missing,
    incomplete, or out-of-sync with the actual trained weights — the old
    config-based path called ``load_state_dict(strict=False)`` which
    silently dropped every unmatched key, leaving LoKr modules at their
    random/zero initialisation and producing garbage outputs.
    """
    try:
        from lycoris import LycorisNetwork
    except ImportError as exc:
        raise ImportError("LyCORIS library not installed. Please install with: pip install lycoris-lora") from exc

    # Clean up any existing LyCORIS network left by a previous load or
    # training run.  Without this, apply_to() wraps already-wrapped layers
    # causing double-injection and completely wrong outputs.
    prev_net = getattr(decoder, "_lycoris_net", None)
    if prev_net is not None:
        try:
            if hasattr(prev_net, "restore"):
                prev_net.restore()
                logger.info("Restored previous LyCORIS network before loading new LoKr adapter")
        except Exception:
            logger.warning("Failed to restore previous LyCORIS network; continuing with best effort")
        try:
            delattr(decoder, "_lycoris_net")
        except Exception:
            pass

    # Primary path: reconstruct network structure from saved weight keys.
    # create_lycoris_from_weights inspects the state-dict keys, matches
    # them to decoder submodules via the LORA_PREFIX naming convention,
    # and rebuilds each LyCORIS module with the correct dim / factor /
    # DoRA settings inferred from the weight tensors themselves.  This is
    # strictly more robust than the old config-based path because it needs
    # no metadata at all.
    try:
        from lycoris import create_lycoris_from_weights

        lycoris_net, _weights_sd = create_lycoris_from_weights(
            1.0, weights_path, decoder,
        )
        n_modules = len(lycoris_net.loras)
        if n_modules == 0:
            logger.warning(
                f"No LoKr modules matched decoder from {weights_path} — "
                "adapter will have no effect"
            )
        else:
            logger.info(f"LoKr adapter: {n_modules} modules reconstructed from weights")
            # Diagnostic: verify weights are actually non-zero
            total_params = 0
            nonzero_params = 0
            for lora_mod in lycoris_net.loras[:3]:  # sample first 3
                for pname, p in lora_mod.named_parameters():
                    total_params += 1
                    if p.abs().sum().item() > 0:
                        nonzero_params += 1
                    else:
                        logger.warning(f"LoKr param all-zero: {pname} shape={tuple(p.shape)}")
            logger.info(
                f"LoKr weight check: {nonzero_params}/{total_params} sampled params are non-zero"
            )
        lycoris_net.apply_to()
        decoder._lycoris_net = lycoris_net
        return lycoris_net
    except Exception:
        logger.warning(
            "create_lycoris_from_weights unavailable or failed; "
            "falling back to config-based loading",
            exc_info=True,
        )

    # Fallback: config-based loading (original approach).
    from lycoris import create_lycoris

    lokr_config = _load_lokr_config(weights_path)
    logger.info(f"Fallback LoKr config from metadata: {dataclasses.asdict(lokr_config)}")
    LycorisNetwork.apply_preset(
        {
            "unet_target_name": lokr_config.target_modules,
            "target_name": lokr_config.target_modules,
        }
    )
    lycoris_net = create_lycoris(
        decoder,
        1.0,
        linear_dim=lokr_config.linear_dim,
        linear_alpha=lokr_config.linear_alpha,
        algo="lokr",
        factor=lokr_config.factor,
        decompose_both=lokr_config.decompose_both,
        use_tucker=lokr_config.use_tucker,
        use_scalar=lokr_config.use_scalar,
        full_matrix=lokr_config.full_matrix,
        bypass_mode=lokr_config.bypass_mode,
        rs_lora=lokr_config.rs_lora,
        unbalanced_factorization=lokr_config.unbalanced_factorization,
    )

    if lokr_config.weight_decompose:
        try:
            lycoris_net = create_lycoris(
                decoder,
                1.0,
                linear_dim=lokr_config.linear_dim,
                linear_alpha=lokr_config.linear_alpha,
                algo="lokr",
                factor=lokr_config.factor,
                decompose_both=lokr_config.decompose_both,
                use_tucker=lokr_config.use_tucker,
                use_scalar=lokr_config.use_scalar,
                full_matrix=lokr_config.full_matrix,
                bypass_mode=lokr_config.bypass_mode,
                rs_lora=lokr_config.rs_lora,
                unbalanced_factorization=lokr_config.unbalanced_factorization,
                dora_wd=True,
            )
        except Exception as exc:
            logger.warning(f"DoRA mode not supported in current LyCORIS build: {exc}")

    # Load weights BEFORE apply_to() — if loading fails (e.g. architecture
    # mismatch), we must NOT leave forward hooks injected on the decoder.
    load_result = lycoris_net.load_weights(weights_path)
    if isinstance(load_result, dict):
        missing = load_result.get("missing keys") or load_result.get("missing_keys") or []
        unexpected = load_result.get("unexpected keys") or load_result.get("unexpected_keys") or []
        if missing:
            logger.warning(f"LoKr load_weights missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"LoKr load_weights unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    lycoris_net.apply_to()
    decoder._lycoris_net = lycoris_net
    _loras = getattr(lycoris_net, "loras", None)
    n_fallback = len(_loras) if isinstance(_loras, (list, tuple)) else 0
    logger.info(f"LoKr adapter loaded via fallback config path ({n_fallback} modules)")
    return lycoris_net


def _default_adapter_name_from_path(lora_path: str) -> str:
    """Derive a default adapter name from path (e.g. 'final' from './lora/final')."""
    name = os.path.basename(lora_path.rstrip(os.sep))
    return name if name else "default"


def add_lora(self, lora_path: str, adapter_name: str | None = None) -> str:
    """Load a LoRA adapter into the decoder under the given name.

    If the decoder is not yet a PeftModel, wraps it and loads the first adapter.
    If it is already a PeftModel, loads an additional adapter (no base restore).
    """
    if self.model is None:
        return "❌ Model not initialized. Please initialize service first."

    if self.quantization is not None:
        logger.warning(
            f"⚠️ Loading LoRA on quantized model ({self.quantization}) — "
            "this may fail or produce unexpected results. INT8 usually works, INT4/NF4 are risky."
        )

    # Warn when loading adapters on XL (4B) models — only XL-trained adapters are compatible
    from acestep.gpu_config import is_xl_model
    config_path = (getattr(self, "last_init_params", None) or {}).get("config_path", "")
    if is_xl_model(config_path):
        logger.warning(
            f"⚠️ Loading adapter on XL model ({config_path}). "
            "Only adapters trained on the XL (4B) architecture will work — "
            "standard 2B adapters have incompatible dimensions and will fail to load."
        )

    if not lora_path or not lora_path.strip():
        return "❌ Please provide a LoRA path."

    lora_path = lora_path.strip()
    if not os.path.exists(lora_path):
        return f"❌ LoRA path not found: {lora_path}"

    lokr_weights_path = _resolve_lokr_weights_path(lora_path)
    _bare_peft_tmp_dir = None  # Track temp dir for cleanup
    if lokr_weights_path is None:
        # If user selected a .safetensors file directly (e.g. via file picker),
        # check whether the parent directory is a PEFT adapter directory.
        if os.path.isfile(lora_path):
            parent = os.path.dirname(lora_path)
            if os.path.exists(os.path.join(parent, "adapter_config.json")):
                logger.info(
                    f"Redirecting file selection to parent PEFT adapter dir: "
                    f"{lora_path} → {parent}"
                )
                lora_path = parent

        config_file = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_file):
            # Fallback: check for bare PEFT LoRA safetensors (e.g. ComfyUI exports)
            # that have lora_down/lora_up keys but no adapter_config.json.
            bare_st = lora_path if os.path.isfile(lora_path) else None
            if bare_st is None and os.path.isdir(lora_path):
                # Pick first .safetensors in the directory
                for name in sorted(os.listdir(lora_path)):
                    if name.lower().endswith(".safetensors"):
                        bare_st = os.path.join(lora_path, name)
                        break
            if bare_st:
                _bare_peft_tmp_dir = _try_prepare_bare_peft_safetensors(bare_st)
            if _bare_peft_tmp_dir:
                logger.info(f"Using inferred PEFT config from bare safetensors: {bare_st}")
                lora_path = _bare_peft_tmp_dir
            else:
                return (
                    "❌ Invalid adapter: expected PEFT LoRA directory containing adapter_config.json "
                    f"or LoKr artifact {LOKR_WEIGHTS_FILENAME} in {lora_path}"
                )

    try:
        from peft import PeftModel
    except ImportError:
        if lokr_weights_path is None:
            return "❌ PEFT library not installed. Please install with: pip install peft"
        PeftModel = None  # type: ignore[assignment]

    effective_name = adapter_name.strip() if isinstance(adapter_name, str) and adapter_name.strip() else _default_adapter_name_from_path(lora_path)
    _active_loras = getattr(self, "_active_loras", None)
    if _active_loras is None:
        self._active_loras = {}
        _active_loras = self._active_loras
    if effective_name in _active_loras:
        return f"❌ Adapter name already in use: {effective_name}. Use a different name or remove it first."

    # ── MERGE MODE: bake adapter weights into base model (zero VRAM overhead) ──
    if getattr(self, "_adapter_merge_mode", False):
        try:
            from acestep.core.generation.handler.lora.advanced_adapter_mixin import (
                _extract_adapter_delta,
                _apply_merged_weights,
                _dequantize_decoder_nf4,
                _requantize_decoder_nf4,
            )

            # If NF4 quantized, dequantize first so backup and adapter ops work
            _needs_nf4_requant = False
            if getattr(self, 'quantization', None) == 'nf4':
                deq_count = _dequantize_decoder_nf4(self.model)
                _needs_nf4_requant = deq_count > 0

            # Backup base decoder on first load (same as advanced system)
            if self._base_decoder is None:
                logger.info("[Merge mode] Backing up base decoder state_dict to CPU")
                backup = {}
                for k, v in self.model.decoder.state_dict().items():
                    if hasattr(v, 'dequantize'):
                        backup[k] = v.dequantize().detach().cpu().clone()
                    else:
                        backup[k] = v.detach().cpu().clone()
                self._base_decoder = backup
                del backup
                _flush_vram("merge-post-backup")
                backup_mb = sum(v.numel() * v.element_size() for v in self._base_decoder.values()) / (1024**2)
                logger.info(f"[Merge mode] Base decoder backed up ({backup_mb:.1f}MB)")

            logger.info(f"[Merge mode] Extracting adapter delta from {lora_path}")
            result = _extract_adapter_delta(self, lora_path)
            adapter_type = result["type"]
            type_label = "LoRA" if adapter_type == "peft_lora" else "LoKr"

            # Extract trigger word metadata
            safetensors_file = result.get("safetensors_file") or lokr_weights_path
            tw, tp = "", ""
            if safetensors_file:
                tw, tp = _read_trigger_word_from_safetensors(safetensors_file)

            # Store in merged basic adapters registry
            self._merged_basic_adapters[effective_name] = {
                "path": lora_path,
                "delta": result["delta"],
                "scale": 1.0,
                "type": adapter_type,
                "trigger_word": tw or "",
                "tag_position": tp or "prepend",
            }

            # Copy into advanced system's _adapter_slots so _apply_merged_weights works
            slot_id = self._next_slot_id
            self._next_slot_id += 1
            self._adapter_slots[slot_id] = {
                "path": lora_path,
                "name": effective_name,
                "type": adapter_type,
                "delta": result["delta"],
                "scale": 1.0,
                "group_scales": {"self_attn": 1.0, "cross_attn": 1.0, "mlp": 1.0, "cond_embed": 1.0},
                "layer_scales": {},
                "trigger_word": tw or "",
                "tag_position": tp or "prepend",
            }

            self.lora_loaded = True
            self.use_lora = True
            self._active_loras[effective_name] = 1.0
            self._adapter_type = "lokr" if adapter_type == "lycoris_lokr" else "lora"
            self._lora_active_adapter = effective_name
            self._merged_dirty = True

            if tw:
                self._adapter_trigger_word = tw
                self._adapter_tag_position = tp or "prepend"
                logger.info(f"Adapter trigger word: '{tw}' (position: {tp or 'prepend'})")
            else:
                self._adapter_trigger_word = ""
                self._adapter_tag_position = ""

            # Apply merged weights to GPU decoder
            _apply_merged_weights(self)

            # Re-quantize merged weights back to NF4 for VRAM savings
            if _needs_nf4_requant:
                _requantize_decoder_nf4(self.model)

            _flush_vram("merge-post-apply")

            delta_keys = len(result["delta"])
            logger.info(
                f"[Merge mode] {type_label} adapter '{effective_name}' merged into base "
                f"({delta_keys} delta keys, slot {slot_id})"
            )
            return f"✅ {type_label} '{effective_name}' merged into model (zero VRAM overhead)"
        except Exception as e:
            logger.exception("[Merge mode] Failed to load adapter")
            return f"❌ Failed to merge adapter: {str(e)}"
        finally:
            if _bare_peft_tmp_dir and os.path.isdir(_bare_peft_tmp_dir):
                try:
                    shutil.rmtree(_bare_peft_tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    # ── PEFT MODE: standard runtime injection ──
    try:  # noqa: SIM117 — need finally for _bare_peft_tmp_dir cleanup
        decoder = self.model.decoder
        is_peft = PeftModel is not None and isinstance(decoder, PeftModel)

        if not is_peft:
            # First LoRA: backup base once, then wrap with PEFT
            if self._base_decoder is None:
                if hasattr(self, "_memory_allocated"):
                    mem_before = self._memory_allocated() / (1024**3)
                    logger.info(f"VRAM before LoRA backup: {mem_before:.2f}GB")
                try:
                    state_dict = decoder.state_dict()
                    if not state_dict:
                        raise ValueError("state_dict is empty - cannot backup decoder")
                    # Dequantize torchao quantized tensors so LoRA merge math works.
                    # Process one tensor at a time to minimise transient VRAM from
                    # dequantize() creating temporary bf16 tensors on GPU.
                    backup = {}
                    for k, v in state_dict.items():
                        if hasattr(v, 'dequantize'):
                            backup[k] = v.dequantize().detach().cpu().clone()
                        else:
                            backup[k] = v.detach().cpu().clone()
                    self._base_decoder = backup
                    del backup  # alias no longer needed
                except Exception as e:
                    logger.error(f"Failed to create state_dict backup: {e}")
                    raise
                finally:
                    # Release state_dict GPU tensor references ASAP — holding them
                    # through PEFT wrapping prevents the CUDA allocator from reusing
                    # those memory blocks and can push 16GB GPUs over the edge.
                    try:
                        del state_dict
                    except NameError:
                        pass
                    _flush_vram("post-backup")
                backup_size_mb = sum(v.numel() * v.element_size() for v in self._base_decoder.values()) / (1024**2)
                logger.info(f"Base decoder state_dict backed up to CPU ({backup_size_mb:.1f}MB)")

            # Flush VRAM before adapter injection — maximise contiguous free blocks
            _flush_vram("pre-adapter-inject")

            if lokr_weights_path is not None:
                logger.info(f"Loading LoKr adapter from {lokr_weights_path} as '{effective_name}'")
                lokr_net = _load_lokr_adapter(decoder, lokr_weights_path)
                n_loaded = len(getattr(lokr_net, 'loras', []) or [])
                if n_loaded == 0:
                    logger.error(f"LoKr adapter loaded 0 modules from {lokr_weights_path}")
                self.model.decoder = decoder
                self._adapter_type = "lokr"
                # Extract trigger word metadata from safetensors header
                tw, tp = _read_trigger_word_from_safetensors(lokr_weights_path)
                if tw:
                    self._adapter_trigger_word = tw
                    self._adapter_tag_position = tp or "prepend"
                    logger.info(f"Adapter trigger word: '{tw}' (position: {tp or 'prepend'})")
                else:
                    self._adapter_trigger_word = ""
                    self._adapter_tag_position = ""
            else:
                logger.info(f"Loading LoRA adapter from {lora_path} as '{effective_name}'")
                self.model.decoder = PeftModel.from_pretrained(
                    decoder, lora_path, adapter_name=effective_name, is_trainable=False
                )
                self._adapter_type = "lora"
        else:
            # Already PEFT: load additional adapter (no base restore). LoKr not supported as second adapter.
            if lokr_weights_path is not None:
                return "❌ LoKr cannot be added as a second adapter when PEFT is already loaded."
            logger.info(f"Loading additional LoRA from {lora_path} as '{effective_name}'")
            self.model.decoder.load_adapter(lora_path, adapter_name=effective_name)
            self._adapter_type = "lora"

        # Flush stale CUDA blocks from adapter injection before casting
        _flush_vram("post-adapter-inject")
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()
        _flush_vram("post-adapter-cast")

        if hasattr(self, "_memory_allocated"):
            mem_after = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM after LoRA load: {mem_after:.2f}GB")

        self.lora_loaded = True
        self.use_lora = True
        self._active_loras[effective_name] = 1.0
        self._ensure_lora_registry()
        self._lora_active_adapter = None
        target_count, adapters = self._rebuild_lora_registry(lora_path=lora_path)
        # Set the newly added adapter as active
        if effective_name in (getattr(self._lora_service, "registry", {}) or {}):
            self._lora_service.set_active_adapter(effective_name)
            self._lora_active_adapter = effective_name
        if hasattr(self.model.decoder, "set_adapter"):
            try:
                self.model.decoder.set_adapter(effective_name)
            except Exception:
                pass

        logger.info(
            f"LoRA adapter '{effective_name}' loaded from {lora_path} "
            f"(adapters={adapters}, targets={target_count})"
        )
        debug_log(
            lambda: f"LoRA registry snapshot: {self._debug_lora_registry_snapshot()}",
            mode=DEBUG_MODEL_LOADING,
            prefix="lora",
        )
        return f"✅ LoRA '{effective_name}' loaded from {lora_path}"
    except Exception as e:
        logger.exception("Failed to load LoRA adapter")
        return f"❌ Failed to load LoRA: {str(e)}"
    finally:
        # Clean up temporary PEFT directory created for bare safetensors
        if _bare_peft_tmp_dir and os.path.isdir(_bare_peft_tmp_dir):
            try:
                shutil.rmtree(_bare_peft_tmp_dir, ignore_errors=True)
            except Exception:
                pass


def load_lora(self, lora_path: str) -> str:
    """Load a single adapter (backward-compat), including LyCORIS LoKr paths."""
    lokr_weights_path = _resolve_lokr_weights_path(lora_path.strip()) if isinstance(lora_path, str) else None
    message = self.add_lora(lora_path, adapter_name=None)
    if lokr_weights_path is not None and message.startswith("✅"):
        return f"✅ LoKr loaded from {lokr_weights_path}"
    return message


def add_voice_lora(self, lora_path: str, scale: float = 1.0) -> str:
    """Load a LoRA as the 'voice' adapter and set its scale. Same machinery as style LoRA."""
    msg = self.add_lora(lora_path, adapter_name="voice")
    if not msg.startswith("✅"):
        return msg
    return self.set_lora_scale("voice", scale)


def remove_lora(self, adapter_name: str) -> str:
    """Remove one LoRA adapter by name. If no adapters remain, restores base decoder."""
    if not self.lora_loaded:
        return "⚠️ No LoRA adapter loaded."

    _active_loras = getattr(self, "_active_loras", None) or {}
    if adapter_name not in _active_loras:
        return f"❌ Unknown adapter: {adapter_name}. Loaded: {list(_active_loras.keys())}"

    # ── MERGE MODE: remove from registry and re-merge remaining ──
    if getattr(self, "_adapter_merge_mode", False) and adapter_name in getattr(self, "_merged_basic_adapters", {}):
        try:
            from acestep.core.generation.handler.lora.advanced_adapter_mixin import _apply_merged_weights

            del self._merged_basic_adapters[adapter_name]
            _active_loras.pop(adapter_name, None)

            # Remove corresponding slot from advanced system
            slot_to_remove = None
            for sid, s in self._adapter_slots.items():
                if s.get("name") == adapter_name:
                    slot_to_remove = sid
                    break
            if slot_to_remove is not None:
                del self._adapter_slots[slot_to_remove]

            if not self._merged_basic_adapters:
                self.lora_loaded = False
                self.use_lora = False
                self._adapter_type = None
                self._adapter_trigger_word = ""
                self._adapter_tag_position = ""

            self._merged_dirty = True
            _apply_merged_weights(self)  # re-merge remaining (or restore base)
            _flush_vram("merge-post-remove")

            logger.info(f"[Merge mode] Adapter '{adapter_name}' removed")
            return f"✅ Adapter '{adapter_name}' removed (merge mode)"
        except Exception as e:
            logger.exception("[Merge mode] Failed to remove adapter")
            return f"❌ Failed to remove adapter: {str(e)}"

    # ── PEFT MODE ──
    try:
        from peft import PeftModel
    except ImportError:
        return "❌ PEFT library not installed."

    decoder = getattr(self.model, "decoder", None)
    if decoder is None or not isinstance(decoder, PeftModel):
        # Inconsistent state: clear our bookkeeping
        _active_loras.pop(adapter_name, None)
        if not _active_loras:
            self.lora_loaded = False
            self.use_lora = False
            self._adapter_type = None
        return "⚠️ Adapter removed from registry (decoder was not PEFT)."

    if adapter_name not in (getattr(decoder, "peft_config", None) or {}):
        _active_loras.pop(adapter_name, None)
        self._ensure_lora_registry()
        self._rebuild_lora_registry()
        return f"✅ Adapter '{adapter_name}' removed (was not in PEFT)."

    try:
        decoder.delete_adapter(adapter_name)
        _active_loras.pop(adapter_name, None)
        remaining = list(_active_loras.keys())
        _flush_vram("post-delete-adapter")

        if not remaining:
            # No adapters left: restore base decoder
            if self._base_decoder is None:
                self.lora_loaded = False
                self.use_lora = False
                self._adapter_type = None
                self._active_loras.clear()
                self._ensure_lora_registry()
                self._lora_service.registry = {}
                self._lora_service.scale_state = {}
                self._lora_service.active_adapter = None
                self._lora_service.last_scale_report = {}
                self._lora_adapter_registry = {}
                self._lora_active_adapter = None
                self._lora_scale_state = {}
                _flush_vram("post-remove-no-backup")
                return "✅ Last adapter removed; base decoder still wrapped (no backup). Restart or load a new LoRA."
            mem_before = None
            if hasattr(self, "_memory_allocated"):
                mem_before = self._memory_allocated() / (1024**3)
                logger.info(f"VRAM before LoRA unload: {mem_before:.2f}GB")
            self.model.decoder = decoder.get_base_model()
            _flush_vram("post-get-base-model")
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
            _flush_vram("pre-restore-cast")
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()
            self.lora_loaded = False
            self.use_lora = False
            self._adapter_type = None
            self._active_loras = {}
            self._ensure_lora_registry()
            self._lora_service.registry = {}
            self._lora_service.scale_state = {}
            self._lora_service.active_adapter = None
            self._lora_service.last_scale_report = {}
            self._lora_adapter_registry = {}
            self._lora_active_adapter = None
            self._lora_scale_state = {}
            _flush_vram("post-restore-complete")
            if mem_before is not None and hasattr(self, "_memory_allocated"):
                mem_after = self._memory_allocated() / (1024**3)
                logger.info(f"VRAM after LoRA unload: {mem_after:.2f}GB (freed: {mem_before - mem_after:.2f}GB)")
            logger.info("LoRA unloaded, base decoder restored")
            return "✅ LoRA unloaded, using base model"
        # Else: set another adapter active and rebuild registry
        next_active = remaining[0]
        if hasattr(decoder, "set_adapter"):
            try:
                decoder.set_adapter(next_active)
            except Exception:
                pass
        self._lora_active_adapter = next_active
        self._ensure_lora_registry()
        self._rebuild_lora_registry()
        self._lora_service.set_active_adapter(next_active)
        # Re-apply scale for the now-active adapter
        scale = self._active_loras.get(next_active, 1.0)
        self._apply_scale_to_adapter(next_active, scale)
        logger.info(f"Adapter '{adapter_name}' removed. Active: {next_active}")
        return f"✅ Adapter '{adapter_name}' removed. Active: {next_active}"
    except Exception as e:
        logger.exception("Failed to remove LoRA adapter")
        return f"❌ Failed to remove LoRA: {str(e)}"


def unload_lora(self) -> str:
    """Unload all LoRA adapters and restore base decoder."""
    if not self.lora_loaded:
        return "⚠️ No LoRA adapter loaded."

    if self._base_decoder is None:
        return "❌ Base decoder backup not found. Cannot restore."

    try:
        mem_before = None
        if hasattr(self, "_memory_allocated"):
            mem_before = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM before LoRA unload: {mem_before:.2f}GB")

        # ── MERGE MODE: clear registries and restore base via _apply_merged_weights ──
        if getattr(self, "_adapter_merge_mode", False) and getattr(self, "_merged_basic_adapters", None):
            from acestep.core.generation.handler.lora.advanced_adapter_mixin import _apply_merged_weights

            self._merged_basic_adapters.clear()
            # Clear the slots that were created for merge mode
            self._adapter_slots.clear()
            self._next_slot_id = 0
            self.use_lora = False
            self._merged_dirty = True
            _apply_merged_weights(self)  # restores base weights
            _flush_vram("merge-post-unload")

            self.lora_loaded = False
            self._adapter_type = None
            self.lora_scale = 1.0
            self._adapter_trigger_word = ""
            self._adapter_tag_position = ""
            _active_loras = getattr(self, "_active_loras", None)
            if _active_loras is not None:
                _active_loras.clear()

            if mem_before is not None and hasattr(self, "_memory_allocated"):
                mem_after = self._memory_allocated() / (1024**3)
                logger.info(f"VRAM after LoRA unload: {mem_after:.2f}GB (freed: {mem_before - mem_after:.2f}GB)")

            logger.info("[Merge mode] All adapters unloaded, base decoder restored")
            return "✅ All adapters unloaded, base model restored (merge mode)"

        # ── PEFT MODE ──
        # If this decoder has an attached LyCORIS net, restore original module graph first.
        lycoris_net = getattr(self.model.decoder, "_lycoris_net", None)
        if lycoris_net is not None:
            restore_fn = getattr(lycoris_net, "restore", None)
            if callable(restore_fn):
                logger.info("Restoring decoder structure from LyCORIS adapter")
                restore_fn()
            else:
                logger.warning("Decoder has _lycoris_net but no restore() method; continuing with state_dict restore")
            self.model.decoder._lycoris_net = None
            _flush_vram("post-lycoris-restore")

        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None  # type: ignore[assignment]

        if PeftModel is not None and isinstance(self.model.decoder, PeftModel):
            logger.info("Removing PEFT LoRA layers via merge_and_unload()")
            # merge_and_unload() removes LoRA layers from the module graph entirely.
            # get_base_model() only unwraps the container but leaves LoRA-modified
            # Linear layers (LoraLayer) in place — their forward hooks keep applying
            # the delta, so unload appears to have no effect.
            self.model.decoder = self.model.decoder.merge_and_unload()
            _flush_vram("post-merge-and-unload")
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
        else:
            logger.info("Restoring base decoder from state_dict backup")
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False, assign=True)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")

        _flush_vram("pre-unload-cast")
        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()

        # Re-quantize to NF4 if the model was originally NF4-quantized.
        # Without this, the restored bf16 base decoder stays in full precision
        # (~8GB) instead of NF4 (~3GB), leaking VRAM on model switch.
        if getattr(self, 'quantization', None) == 'nf4':
            from acestep.core.generation.handler.lora.advanced_adapter_mixin import _requantize_decoder_nf4
            _requantize_decoder_nf4(self.model)

        self.lora_loaded = False
        self.use_lora = False
        self._adapter_type = None
        self.lora_scale = 1.0
        self._adapter_trigger_word = ""
        self._adapter_tag_position = ""
        _active_loras = getattr(self, "_active_loras", None)
        if _active_loras is not None:
            _active_loras.clear()
        self._ensure_lora_registry()
        self._lora_service.registry = {}
        self._lora_service.scale_state = {}
        self._lora_service.active_adapter = None
        self._lora_service.last_scale_report = {}
        self._lora_adapter_registry = {}
        self._lora_active_adapter = None
        self._lora_scale_state = {}

        _flush_vram("post-unload-complete")
        if mem_before is not None and hasattr(self, "_memory_allocated"):
            mem_after = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM after LoRA unload: {mem_after:.2f}GB (freed: {mem_before - mem_after:.2f}GB)")

        logger.info("LoRA unloaded, base decoder restored")
        return "✅ LoRA unloaded, using base model"
    except Exception as e:
        logger.exception("Failed to unload LoRA")
        return f"❌ Failed to unload LoRA: {str(e)}"
