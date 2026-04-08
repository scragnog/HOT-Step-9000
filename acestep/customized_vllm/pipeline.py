"""Inference pipeline: model loading, KV cache provisioning, CUDA graphs, and forward passes."""

import torch
import sys

from loguru import logger

from acestep.customized_vllm.transformer import CausalTransformer, load_weights
from acestep.debug_utils import debug_start, debug_end


# ---------------------------------------------------------------------------
# Nucleus / top-k filtering (always receives tensors, never None)
# ---------------------------------------------------------------------------

def _filter_by_top_k(logits, k):
    """Top-k filtering without full vocabulary sort."""
    vocab_size = logits.shape[1]
    skip = (k <= 0) | (k >= vocab_size)
    k_safe = k.masked_fill(skip, 1).long()
    max_k = int(k_safe.max().clamp(max=vocab_size))
    topk_vals = logits.topk(max_k, dim=1).values
    thresh = topk_vals.gather(1, (k_safe - 1).clamp(0, max_k - 1).unsqueeze(1))
    thresh.masked_fill_(skip.unsqueeze(1), float("-inf"))
    logits.masked_fill_(logits < thresh, float("-inf"))
    return logits


def _filter_by_nucleus(logits, k, p):
    """Combined top-k and nucleus (top-p) filtering.

    Parameters are always tensors (never None).
    k=0 means skip top-k, p=1.0 means skip top-p.
    Note: NOT compiled because .any() and int() cause graph breaks in dynamo.
    """
    has_k = (k > 0).any()
    has_p = (p < 1.0).any()
    if not has_p and not has_k:
        return logits
    if not has_p:
        return _filter_by_top_k(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if has_k:
        vocab_size = logits_sort.size(1)
        k_clamped = k.clamp(1, vocab_size).long()
        thresh = logits_sort.gather(1, (vocab_size - k_clamped).unsqueeze(1))
        logits_sort.masked_fill_(logits_sort < thresh, float("-inf"))

    probs_sum = logits_sort.softmax(dim=-1).cumsum_(dim=-1)
    mask = probs_sum <= (1.0 - p.unsqueeze(1))
    mask[:, -1] = False
    logits_sort.masked_fill_(mask, float("-inf"))
    logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def sample_tokens(logits, temperatures, top_ks, top_ps):
    """Temperature-scaled sampling with top-k/top-p filtering.

    Uses Gumbel-max trick for efficient categorical sampling.
    """
    logits = logits.float().div_(temperatures.unsqueeze(1))
    _filter_by_nucleus(logits, top_ks, top_ps)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Loads a model, provisions KV cache, captures CUDA graphs, and runs forward passes."""

    def __init__(self, hf_config, model_path: str, block_size: int, max_num_seqs: int,
                 max_num_batched_tokens: int, max_model_len: int, gpu_memory_utilization: float,
                 enforce_eager: bool):
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.verbose = True
        self.block_size = block_size
        self.enforce_eager = enforce_eager
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.hf_config = hf_config

        torch.cuda.set_device(0)
        saved_dtype = torch.get_default_dtype()

        gpu_props = torch.cuda.get_device_properties(0)
        bf16_ok = (gpu_props.major, gpu_props.minor) >= (8, 0)
        raw = getattr(hf_config, "dtype", getattr(hf_config, "torch_dtype", None))
        if isinstance(raw, str):
            _map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
            raw = _map.get(raw.replace("torch.", ""), None)
        self.dtype = (raw if isinstance(raw, torch.dtype) and raw.is_floating_point else
                      torch.bfloat16 if bf16_ok else torch.float16)
        if self.dtype == torch.bfloat16 and not bf16_ok:
            self.dtype = torch.float16

        torch.set_default_dtype(self.dtype)
        torch.set_default_device("cuda")
        try:
            self.model = CausalTransformer(hf_config)
            _t = debug_start("load_model", prefix="tensor.vllm")
            load_weights(self.model, model_path)
            debug_end("load_model", _t, prefix="tensor.vllm")

            self._init_transfer_buffers()
            self._warmup_pipeline()
            self._provision_kv_storage()
            if not enforce_eager:
                self._compile_execution_graphs()
        finally:
            torch.set_default_device("cpu")
            torch.set_default_dtype(saved_dtype)

    # -- Transfer buffers ------------------------------------------------

    def _init_transfer_buffers(self):
        """Pre-allocate pinned CPU buffers used to shuttle data to GPU."""
        bs = self.max_num_seqs
        max_blocks = (self.max_model_len + self.block_size - 1) // self.block_size
        pin = dict(dtype=torch.float32, device="cpu", pin_memory=True)
        pin_i32 = dict(dtype=torch.int32, device="cpu", pin_memory=True)
        pin_i64 = dict(dtype=torch.int64, device="cpu", pin_memory=True)
        # Dict-based buffer storage (structurally different from per-attribute style)
        self._xfer = {
            "temps": torch.zeros(bs, **pin),
            "guidance": torch.zeros(bs, **pin),
            "top_k": torch.zeros(bs, **pin_i32),
            "top_p": torch.zeros(bs, **pin),
            "rep_pen": torch.zeros(bs, **pin),
            "token_ids": torch.zeros(bs, **pin_i64),
            "positions": torch.zeros(bs, **pin_i64),
            "slots": torch.zeros(bs, **pin_i32),
            "ctx_lens": torch.zeros(bs, **pin_i32),
        }

    # -- Warmup & KV storage ---------------------------------------------

    def _warmup_pipeline(self):
        from acestep.customized_vllm import GenerationSlot, reset_context
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        n = min(self.max_num_batched_tokens // self.max_model_len, self.max_num_seqs)
        dummy_slots = [GenerationSlot([0] * self.max_model_len, block_size=self.block_size) for _ in range(n)]
        self._execute_prefill(dummy_slots)
        reset_context()
        torch.cuda.empty_cache()

    def _provision_kv_storage(self):
        _t = debug_start("allocate_kv_cache", prefix="tensor.vllm")
        hf = self.hf_config
        free, total = torch.cuda.mem_get_info()
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        import os
        sim = os.environ.get("MAX_CUDA_VRAM")
        if sim:
            try:
                cap = float(sim) * 1024**3
                if cap < total:
                    total = int(cap)
                    free = max(0, total - torch.cuda.memory_reserved())
            except (ValueError, TypeError):
                pass

        num_kv_heads = hf.num_key_value_heads
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        block_bytes = 2 * hf.num_hidden_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize

        target = total * self.gpu_memory_utilization
        term_a = free * 0.9
        term_b = target - current
        term_c = max(0, free - 1024**3) * 0.9
        avail = min(term_a, term_b, term_c)
        used_fallback = False
        if avail <= 0:
            avail = free * 0.5
            used_fallback = True

        logger.info(
            f"[customized_vllm] KV budget: "
            f"free={free / 1024**3:.2f}GB, total={total / 1024**3:.2f}GB, "
            f"pytorch_alloc={current / 1024**3:.2f}GB, "
            f"gpu_mem_util={self.gpu_memory_utilization:.3f}, "
            f"target={target / 1024**3:.2f}GB, "
            f"terms=[{term_a / 1024**3:.2f}, {term_b / 1024**3:.2f}, {term_c / 1024**3:.2f}], "
            f"avail={avail / 1024**3:.2f}GB"
            f"{' (FALLBACK: avail was <=0)' if used_fallback else ''}"
        )

        self._num_cache_blocks = max(1, int(avail) // block_bytes)
        cap = self._num_cache_blocks * self.block_size
        gb = self._num_cache_blocks * block_bytes / 1024**3
        logger.info(f"[customized_vllm] KV cache: {self._num_cache_blocks} blocks, "
                    f"{cap} tokens, {gb:.2f} GB")

        self._kv_storage = torch.empty(
            2, hf.num_hidden_layers, self._num_cache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        layer_id = 0
        for m in self.model.modules():
            if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
                m.k_cache = self._kv_storage[0, layer_id]
                m.v_cache = self._kv_storage[1, layer_id]
                layer_id += 1
        debug_end("allocate_kv_cache", _t, prefix="tensor.vllm")

    # -- Input preparation -----------------------------------------------

    def _build_cache_index(self, slots):
        max_len = max(len(s.cache_blocks) for s in slots)
        rows = [s.cache_blocks + [-1] * (max_len - len(s.cache_blocks)) for s in slots]
        return torch.tensor(rows, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def _execute_prefill(self, slots):
        """Prepare prefill inputs and run model forward, returning logits."""
        from acestep.customized_vllm import _set_forward_state
        ids, pos, cu_q, cu_k = [], [], [0], [0]
        max_sq = max_sk = 0
        slot_map = []
        for s in slots:
            n = len(s)
            ids.extend(s.token_ids)
            pos.extend(range(n))
            cu_q.append(cu_q[-1] + n)
            cu_k.append(cu_k[-1] + n)
            max_sq = max(n, max_sq)
            max_sk = max(n, max_sk)
            for i in range(s.required_blocks):
                if not s.cache_blocks:
                    continue
                start = s.cache_blocks[i] * self.block_size
                end = start + (s.tail_block_fill if i == s.required_blocks - 1 else self.block_size)
                slot_map.extend(range(start, end))

        ids = torch.tensor(ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        pos = torch.tensor(pos, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_q = torch.tensor(cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k = torch.tensor(cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(slot_map, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        _set_forward_state(True, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                           max_seqlen_q=max_sq, max_seqlen_k=max_sk, slot_mapping=sm)
        return self._forward_pass(ids, pos, is_prefill=True)

    def _execute_autoregressive(self, slots):
        """Prepare single-token decode inputs and run model forward, returning logits."""
        from acestep.customized_vllm import _set_forward_state
        bs = len(slots)
        xfer = self._xfer
        for i, s in enumerate(slots):
            xfer["token_ids"][i] = s.last_token
            xfer["positions"][i] = len(s) - 1
            xfer["ctx_lens"][i] = len(s)
            xfer["slots"][i] = s.cache_blocks[-1] * self.block_size + s.tail_block_fill - 1

        ids = xfer["token_ids"][:bs].cuda(non_blocking=True)
        pos = xfer["positions"][:bs].cuda(non_blocking=True)
        sm = xfer["slots"][:bs].cuda(non_blocking=True)
        cl = xfer["ctx_lens"][:bs].cuda(non_blocking=True)
        bt = self._build_cache_index(slots)
        _set_forward_state(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
        return self._forward_pass(ids, pos, is_prefill=False)

    # -- Model forward ---------------------------------------------------

    @torch.inference_mode()
    def _forward_pass(self, input_ids, positions, is_prefill):
        from acestep.customized_vllm import _get_forward_state
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.project_to_vocab(self.model(input_ids, positions))

        bs = input_ids.size(0)
        state = _get_forward_state()
        gio = self._graph_io
        max_cols = gio["block_tables"].size(1)
        if (state.block_tables.size(1) > max_cols or state.block_tables.size(0) != bs
                or state.slot_mapping.size(0) != bs or state.context_lens.size(0) != bs):
            return self.model.project_to_vocab(self.model(input_ids, positions))

        graph = self._graphs[next(x for x in self._compiled_sizes if x >= bs)]
        gio["input_ids"][:bs] = input_ids
        gio["positions"][:bs] = positions
        gio["slot_mapping"].fill_(-1)
        gio["slot_mapping"][:bs] = state.slot_mapping
        gio["context_lens"].zero_()
        gio["context_lens"][:bs] = state.context_lens
        gio["block_tables"][:bs].fill_(-1)
        gio["block_tables"][:bs, :state.block_tables.size(1)] = state.block_tables
        graph.replay()
        return self.model.project_to_vocab(gio["outputs"][:bs])

    # -- Sampling helpers ------------------------------------------------

    def _gather_sampling_config(self, slots, is_cfg):
        """Pack per-slot sampling parameters into GPU tensors.

        Always returns tensors (never None) so that `sample_tokens` receives
        a stable signature for torch.compile.
        """
        targets = slots[:len(slots) // 2] if is_cfg else slots
        n = len(targets)
        xfer = self._xfer
        for i, s in enumerate(targets):
            xfer["temps"][i] = s.temperature
            xfer["guidance"][i] = s.cfg_scale
            xfer["top_k"][i] = s.top_k if s.top_k else 0
            xfer["top_p"][i] = s.top_p if s.top_p else 1.0
            xfer["rep_pen"][i] = s.repetition_penalty if s.repetition_penalty else 1.0
        return (
            xfer["temps"][:n].cuda(non_blocking=True),
            xfer["guidance"][:n].cuda(non_blocking=True),
            xfer["top_k"][:n].cuda(non_blocking=True),    # always a tensor
            xfer["top_p"][:n].cuda(non_blocking=True),     # always a tensor
            xfer["rep_pen"][:n].cuda(non_blocking=True),
        )

    def _constrain_logits(self, logits, slots):
        """Apply per-slot logits processors.

        Each slot with a non-None logits_processor is processed individually
        using its own token history.  In typical ACE-Step usage all batch
        slots share the same processor instance, but this handles the general
        case correctly.
        """
        if not slots:
            return logits
        try:
            for i, slot in enumerate(slots):
                if slot.logits_processor is None:
                    continue
                ids_t = torch.tensor([slot.token_ids], device=logits.device)
                processed = slot.logits_processor(ids_t, logits[i:i+1].clone())
                logits[i] = processed[0]
        except TypeError:
            import traceback
            logger.error(f"TypeError in _constrain_logits "
                         f"(n_slots={len(slots)}, slot_idx={i}, "
                         f"processor_state={getattr(slot.logits_processor, 'state', '?')}):\n"
                         f"{traceback.format_exc()}")
            raise
        return logits

    def _penalize_repetitions(self, logits, slots, penalties):
        if penalties is None:
            return logits
        for i, slot in enumerate(slots):
            p = penalties[i].item()
            if p == 1.0:
                continue
            comp = torch.tensor(slot.generated_ids, device=logits.device)
            if len(comp) == 0:
                continue
            mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            mask[comp] = True
            penalized = torch.where(logits[i] < 0, logits[i] * p, logits[i] / p)
            logits[i] = torch.where(mask, penalized, logits[i])
        return logits

    # -- Main step -------------------------------------------------------

    def execute_step(self, slots, is_prefill):
        """Full forward + sampling step. Returns list of sampled token IDs."""
        from acestep.customized_vllm import reset_context
        try:
            is_cfg = slots[0].cfg_scale > 1.0 and slots[0].paired_slot is not None
            try:
                logits = (self._execute_prefill(slots) if is_prefill
                          else self._execute_autoregressive(slots))
            finally:
                reset_context()
            # _forward_pass runs under @torch.inference_mode() so the returned
            # logits are "inference tensors" that block in-place mutation.
            # Clone to get regular tensors before _penalize_repetitions etc.
            logits = logits.clone()
            temps, cfg_s, topk, topp, rep_pen = self._gather_sampling_config(slots, is_cfg)

            if is_cfg:
                nc = len(slots) // 2
                cond, uncond = logits[:nc], logits[nc:]
                cond = self._penalize_repetitions(cond, slots[:nc], rep_pen)
                cfg_logits = uncond + cfg_s.unsqueeze(1) * (cond - uncond)
                cfg_logits = torch.nan_to_num(cfg_logits, nan=float('-inf'))
                cfg_logits = self._constrain_logits(cfg_logits, slots[:nc])
                tids = sample_tokens(cfg_logits, temps, topk, topp).tolist()
                for i, s in enumerate(slots[:nc]):
                    if s.logits_processor_update_state:
                        s.logits_processor_update_state(tids[i])
                return tids

            logits = self._penalize_repetitions(logits, slots, rep_pen)
            logits = self._constrain_logits(logits.clone(), slots)
            tids = sample_tokens(logits, temps, topk, topp).tolist()
            for i, s in enumerate(slots):
                if s.logits_processor_update_state:
                    s.logits_processor_update_state(tids[i])
            return tids
        except TypeError:
            import traceback
            logger.error(f"TypeError in execute_step "
                         f"(prefill={is_prefill}, n_slots={len(slots)}):\n"
                         f"{traceback.format_exc()}")
            raise

    # -- CUDA graph capture ----------------------------------------------

    @torch.inference_mode()
    def _compile_execution_graphs(self):
        from acestep.customized_vllm import _set_forward_state, reset_context
        _t = debug_start("capture_cudagraph", prefix="tensor.vllm")
        max_bs = min(self.max_num_seqs, 512)
        max_blocks = (self.max_model_len + self.block_size - 1) // self.block_size
        ids = torch.zeros(max_bs, dtype=torch.int64)
        pos = torch.zeros(max_bs, dtype=torch.int64)
        sm = torch.zeros(max_bs, dtype=torch.int32)
        cl = torch.zeros(max_bs, dtype=torch.int32)
        bt = torch.zeros(max_bs, max_blocks, dtype=torch.int32)
        out = torch.zeros(max_bs, self.hf_config.hidden_size)
        self._compiled_sizes = sorted(set([1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) + [max_bs]))
        self._graphs = {}
        pool = None
        for bs in reversed(self._compiled_sizes):
            g = torch.cuda.CUDAGraph()
            _set_forward_state(False, slot_mapping=sm[:bs], context_lens=cl[:bs], block_tables=bt[:bs])
            out[:bs] = self.model(ids[:bs], pos[:bs])
            with torch.cuda.graph(g, pool):
                out[:bs] = self.model(ids[:bs], pos[:bs])
            if pool is None:
                pool = g.pool()
            self._graphs[bs] = g
            torch.cuda.synchronize()
            reset_context()
        self._graph_io = dict(input_ids=ids, positions=pos, slot_mapping=sm,
                              context_lens=cl, block_tables=bt, outputs=out)
        debug_end("capture_cudagraph", _t, prefix="tensor.vllm")

    def shutdown(self):
        if not self.enforce_eager:
            del self._graphs, self._graph_io
        torch.cuda.synchronize()
