"""Customised vLLM inference engine for single-GPU Qwen3 generation.

Public API:
    LLM            - High-level generate() interface with optional CFG
    SamplingParams - Per-request sampling configuration
    reset_context  - Clear thread-local forward state (call on error recovery)
"""

import os
import atexit
import threading as _threading
from collections import deque
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from time import perf_counter
from typing import Optional, Callable, Any

import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

# ---------------------------------------------------------------------------
# Sampling configuration
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    """Per-request configuration for token sampling."""
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    cfg_scale: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    logits_processor: Optional[Any] = field(default=None, repr=False)
    logits_processor_update_state: Optional[Callable[[int], None]] = field(default=None, repr=False)

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert self.max_tokens > 0, "max_tokens must be > 0"
        assert self.cfg_scale >= 1.0, "cfg_scale must be >= 1.0"
        if self.top_k is not None:
            assert self.top_k > 0, "top_k must be > 0"
        if self.top_p is not None:
            assert 0.0 < self.top_p <= 1.0, "top_p must be in (0.0, 1.0]"
        assert self.repetition_penalty > 0.0, "repetition_penalty must be > 0.0"


# ---------------------------------------------------------------------------
# Thread-local forward state (replaces separate context.py module)
# ---------------------------------------------------------------------------

@dataclass
class ForwardState:
    """Attention metadata shared between the pipeline and transformer layers."""
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_TLS = _threading.local()


def _get_forward_state() -> ForwardState:
    """Retrieve the current thread's forward state."""
    s = getattr(_TLS, "_fwd", None)
    if s is None:
        s = ForwardState()
        _TLS._fwd = s
    return s


def _set_forward_state(is_prefill=False, **kw) -> ForwardState:
    """Replace the current thread's forward state."""
    _TLS._fwd = ForwardState(is_prefill=is_prefill, **kw)
    return _TLS._fwd


def reset_context():
    """Reset forward state to defaults (public, used by llm_inference error recovery)."""
    _TLS._fwd = ForwardState()


# ---------------------------------------------------------------------------
# Generation slot (per-sequence state)
# ---------------------------------------------------------------------------

class SlotStatus(Enum):
    PENDING = auto()
    ACTIVE = auto()
    DONE = auto()


class GenerationSlot:
    """Tracks token state, cache blocks, and sampling config for a single sequence."""

    _id_gen = count()

    def __init__(self, token_ids: list[int], params=SamplingParams(),
                 is_unconditional: bool = False, block_size: int = 256):
        self.slot_id = next(GenerationSlot._id_gen)
        self.block_size = block_size
        self.status = SlotStatus.PENDING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(token_ids)
        self.prompt_length = len(token_ids)
        self.cache_blocks: list[int] = []
        self.temperature = params.temperature
        self.max_tokens = params.max_tokens
        self.ignore_eos = params.ignore_eos
        self.cfg_scale = params.cfg_scale
        self.top_k = params.top_k
        self.top_p = params.top_p
        self.repetition_penalty = params.repetition_penalty
        self.is_unconditional = is_unconditional
        self.paired_slot: Optional["GenerationSlot"] = None
        self.logits_processor: Optional[Any] = params.logits_processor
        self.logits_processor_update_state: Optional[Callable[[int], None]] = (
            params.logits_processor_update_state
        )

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SlotStatus.DONE

    @property
    def generated_ids(self):
        return self.token_ids[self.prompt_length:]

    @property
    def required_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def tail_block_fill(self):
        return self.num_tokens - (self.required_blocks - 1) * self.block_size

    def push_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


# ---------------------------------------------------------------------------
# KV cache pool
# ---------------------------------------------------------------------------

class CachePool:
    """Simple block-based KV cache allocator (no prefix caching)."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.available: deque[int] = deque(range(num_blocks))
        self.total = num_blocks

    def has_capacity(self, num_blocks: int) -> bool:
        return len(self.available) >= num_blocks

    def reserve(self, slot: GenerationSlot):
        for _ in range(slot.required_blocks):
            slot.cache_blocks.append(self.available.popleft())

    def release(self, slot: GenerationSlot):
        for bid in reversed(slot.cache_blocks):
            self.available.append(bid)
        slot.cache_blocks.clear()

    def grow_if_needed(self, slot: GenerationSlot):
        if len(slot) % self.block_size == 1 and len(slot) > slot.prompt_length:
            if not self.available:
                raise RuntimeError(
                    f"KV cache exhausted during decode: slot {slot.slot_id} needs a new block "
                    f"but 0/{self.total} blocks are available"
                )
            slot.cache_blocks.append(self.available.popleft())


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@dataclass
class _EngineConfig:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    kvcache_block_size: int = 256

    def __post_init__(self):
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len


# ---------------------------------------------------------------------------
# LLM - public inference API
# ---------------------------------------------------------------------------

class LLM:
    """High-level inference engine with optional classifier-free guidance.

    Usage::

        llm = LLM(model="/path/to/model")
        outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=128))
    """

    def __init__(self, model, **kwargs):
        from acestep.customized_vllm.pipeline import InferencePipeline

        cfg = _EngineConfig(
            model=model,
            max_num_batched_tokens=kwargs.get("max_num_batched_tokens", 16384),
            max_num_seqs=kwargs.get("max_num_seqs", 512),
            max_model_len=kwargs.get("max_model_len", 4096),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            enforce_eager=kwargs.get("enforce_eager", False),
            kvcache_block_size=kwargs.get("kvcache_block_size", 256),
        )
        self._cfg = cfg
        self._lock = _threading.Lock()
        self._pipeline = InferencePipeline(
            hf_config=cfg.hf_config, model_path=cfg.model,
            block_size=cfg.kvcache_block_size, max_num_seqs=cfg.max_num_seqs,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enforce_eager=cfg.enforce_eager,
        )
        tok = kwargs.get("tokenizer", None)
        self.tokenizer = tok if tok is not None else AutoTokenizer.from_pretrained(model, use_fast=True)
        self._eos = self.tokenizer.eos_token_id
        self._block_size = cfg.kvcache_block_size
        self._cache = CachePool(self._pipeline._num_cache_blocks, cfg.kvcache_block_size)
        self._active_slots: list[GenerationSlot] = []
        atexit.register(self.exit)

    def exit(self):
        self._pipeline.shutdown()

    def reset(self):
        """Release all KV cache blocks (call on error to prevent leaks)."""
        for slot in self._active_slots:
            if slot.cache_blocks:
                self._cache.release(slot)
        self._active_slots.clear()

    # -- Public generate --------------------------------------------------

    def generate(self, prompts, sampling_params, use_tqdm=True, unconditional_prompts=None):
        """Generate completions for a batch of prompts.

        Returns list of dicts with ``"text"`` and ``"token_ids"`` keys.
        """
        with self._lock:
            return self._run_generation(prompts, sampling_params, use_tqdm, unconditional_prompts)

    # -- Internal generation logic ----------------------------------------

    def _prepare_slots(self, prompts, sampling_params, unconditional_prompts):
        """Tokenise prompts, create slots, allocate KV cache."""
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if unconditional_prompts is None:
            unconditional_prompts = [None] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError(
                f"sampling_params length ({len(sampling_params)}) != prompts length ({len(prompts)})"
            )
        if len(unconditional_prompts) != len(prompts):
            raise ValueError(
                f"unconditional_prompts length ({len(unconditional_prompts)}) != prompts length ({len(prompts)})"
            )

        all_slots = []
        for prompt, sp, uncond in zip(prompts, sampling_params, unconditional_prompts):
            ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
            bs = getattr(self, '_block_size', 256)
            cond = GenerationSlot(ids, sp, block_size=bs)
            if sp.cfg_scale > 1.0:
                u_ids = (self.tokenizer.encode(uncond) if isinstance(uncond, str)
                         else (uncond if uncond is not None else ids))
                uncond_slot = GenerationSlot(u_ids, sp, is_unconditional=True, block_size=bs)
                cond.paired_slot = uncond_slot
                uncond_slot.paired_slot = cond
                all_slots.extend([cond, uncond_slot])
            else:
                all_slots.append(cond)

        total_blocks = sum(s.required_blocks for s in all_slots)
        if not self._cache.has_capacity(total_blocks):
            raise RuntimeError(
                f"Insufficient KV cache: need {total_blocks} blocks, "
                f"have {len(self._cache.available)}/{self._cache.total}"
            )
        for slot in all_slots:
            self._cache.reserve(slot)
            slot.status = SlotStatus.ACTIVE
        self._active_slots = list(all_slots)
        return all_slots

    def _arrange_guidance_batch(self, slots):
        """Order: normal, then CFG conditional, then CFG unconditional."""
        normal = [s for s in slots if s.cfg_scale <= 1.0]
        cond = [s for s in slots if s.cfg_scale > 1.0 and not s.is_unconditional]
        uncond = [s for s in slots if s.is_unconditional]
        return normal + cond + uncond

    def _generation_steps(self, all_slots):
        """Generator yielding (ordered_batch, token_ids, elapsed, is_prefill) per step."""
        ordered = self._arrange_guidance_batch(all_slots)
        t = perf_counter()
        tids = self._pipeline.execute_step(ordered, is_prefill=True)
        yield ordered, tids, perf_counter() - t, True

        while self._active_slots:
            for slot in self._active_slots:
                self._cache.grow_if_needed(slot)
            batch = self._arrange_guidance_batch(self._active_slots)
            t = perf_counter()
            tids = self._pipeline.execute_step(batch, is_prefill=False)
            yield batch, tids, perf_counter() - t, False

    def _finalize_step(self, batch, token_ids, outputs, pbar):
        """Append tokens, check EOS, collect finished outputs."""
        is_cfg = (len(batch) > 0 and batch[0].cfg_scale > 1.0
                  and batch[0].paired_slot is not None)
        if is_cfg:
            nc = len(batch) // 2
            for cond, uncond, tid in zip(batch[:nc], batch[nc:], token_ids):
                cond.push_token(tid)
                uncond.push_token(tid)
                done = ((not cond.ignore_eos and tid == self._eos) or
                        cond.num_tokens - cond.prompt_length >= cond.max_tokens)
                if done:
                    for s in (cond, uncond):
                        s.status = SlotStatus.DONE
                        self._cache.release(s)
                        if s in self._active_slots:
                            self._active_slots.remove(s)
                    outputs[cond.slot_id] = cond.generated_ids
                    if pbar:
                        pbar.update(1)
        else:
            for slot, tid in zip(batch, token_ids):
                slot.push_token(tid)
                done = ((not slot.ignore_eos and tid == self._eos) or
                        slot.num_tokens - slot.prompt_length >= slot.max_tokens)
                if done:
                    slot.status = SlotStatus.DONE
                    self._cache.release(slot)
                    if slot in self._active_slots:
                        self._active_slots.remove(slot)
                    outputs[slot.slot_id] = slot.generated_ids
                    if pbar:
                        pbar.update(1)

    def _run_generation(self, prompts, sampling_params, use_tqdm, unconditional_prompts):
        if self._active_slots:
            self.reset()

        all_slots = self._prepare_slots(prompts, sampling_params, unconditional_prompts)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None
        prefill_tps = decode_tps = 0.0
        outputs = {}

        try:
            for batch, token_ids, elapsed, is_pf in self._generation_steps(all_slots):
                elapsed = max(elapsed, 1e-9)
                if is_pf:
                    prefill_tps = sum(len(s) for s in batch) / elapsed
                else:
                    n_cond = sum(1 for s in batch if not s.is_unconditional)
                    decode_tps = n_cond / elapsed

                self._finalize_step(batch, token_ids, outputs, pbar)
                if pbar:
                    pbar.set_postfix(Prefill=f"{int(prefill_tps)}tok/s",
                                     Decode=f"{int(decode_tps)}tok/s")
                if not self._active_slots:
                    break
        except Exception:
            self.reset()
            raise
        finally:
            if pbar:
                pbar.close()

        result = [outputs[sid] for sid in sorted(outputs)]
        return [{"text": self.tokenizer.decode(tids), "token_ids": tids} for tids in result]
