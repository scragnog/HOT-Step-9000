"""
Adapter Adherence Sweep Tool

Systematically tests different adapter scale/group-scale configurations
and scores each generation for lyric adherence using:
1. DiT alignment score (from API)
2. LRC timestamp analysis
3. Whisper transcription diffing

Usage:
    .venv\\Scripts\\python.exe scripts/adherence_sweep.py --config settingstouse.json
    .venv\\Scripts\\python.exe scripts/adherence_sweep.py --config settingstouse.json --dry-run
    .venv\\Scripts\\python.exe scripts/adherence_sweep.py --config settingstouse.json --max-configs 5
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adherence_scoring import (
    AdherenceResult,
    WhisperScorer,
    compute_adherence_score,
    extract_lyric_lines,
    score_lrc_timing,
)

# ── API Configuration ─────────────────────────────────────────────────────

API_BASE = "http://localhost:8001"
API_HEADERS = {"Content-Type": "application/json"}


# ── Sweep Configuration ──────────────────────────────────────────────────


@dataclass
class SweepConfig:
    """A single parameter configuration to test."""

    config_id: str
    adapter_scale: float
    self_attn: float
    cross_attn: float
    mlp: float
    thinking: bool
    label: str = ""

    def summary(self) -> str:
        t = "T" if self.thinking else "F"
        return f"scale={self.adapter_scale:.1f} SA={self.self_attn:.2f} CA={self.cross_attn:.2f} MLP={self.mlp:.1f} think={t}"


def build_sweep_configs(
    base_scale: float = 0.7,
    base_sa: float = 0.35,
    base_ca: float = 0.35,
    base_mlp: float = 1.8,
) -> List[SweepConfig]:
    """Build the default sweep configuration set.

    Tests each parameter independently against the baseline,
    plus key diagnostic combos and both thinking modes.
    """
    configs = []
    idx = 0

    def _add(scale, sa, ca, mlp, thinking, label):
        nonlocal idx
        idx += 1
        configs.append(
            SweepConfig(
                config_id=f"{idx:03d}",
                adapter_scale=scale,
                self_attn=sa,
                cross_attn=ca,
                mlp=mlp,
                thinking=thinking,
                label=label,
            )
        )

    # ── Reference: no adapter ──
    _add(0.0, 0.0, 0.0, 0.0, True, "No adapter + thinking")
    _add(0.0, 0.0, 0.0, 0.0, False, "No adapter + no thinking")

    # ── Baseline (user's current settings) ──
    _add(base_scale, base_sa, base_ca, base_mlp, True, "Baseline + thinking")
    _add(base_scale, base_sa, base_ca, base_mlp, False, "Baseline + no thinking")

    # ── Overall scale sweep (other params at baseline) ──
    for s in [0.3, 0.5, 1.0]:
        if s != base_scale:
            _add(s, base_sa, base_ca, base_mlp, True, f"Scale={s}")

    # ── Self-attention sweep ──
    for sa in [0.0, 0.7, 1.0]:
        if sa != base_sa:
            _add(base_scale, sa, base_ca, base_mlp, True, f"SA={sa}")

    # ── Cross-attention sweep ──
    for ca in [0.0, 0.7, 1.0]:
        if ca != base_ca:
            _add(base_scale, base_sa, ca, base_mlp, True, f"CA={ca}")

    # ── MLP sweep ──
    for mlp in [0.5, 1.0, 1.5]:
        if mlp != base_mlp:
            _add(base_scale, base_sa, base_ca, mlp, True, f"MLP={mlp}")

    # ── Diagnostic combos: isolate attention components ──
    _add(base_scale, 0.0, 0.0, base_mlp, True, "MLP-only (SA=0, CA=0)")
    _add(base_scale, 0.0, 0.0, base_mlp, False, "MLP-only + no thinking")
    _add(base_scale, 1.0, 0.0, 1.0, True, "No cross-attn (CA=0)")
    _add(base_scale, 0.0, 1.0, 1.0, True, "No self-attn (SA=0)")

    # ── Optimised guesses ──
    _add(0.5, 0.2, 0.2, 1.5, True, "Conservative blend")
    _add(0.5, 0.0, 0.0, 1.5, True, "Conservative MLP-only")
    _add(base_scale, 0.0, 0.0, 1.0, True, "Clean MLP=1.0 only")

    return configs


# ── API Client ────────────────────────────────────────────────────────────


def set_adapter_scale(scale: float, slot: int = 0) -> bool:
    """Set the overall adapter scale via the API."""
    try:
        resp = requests.post(
            f"{API_BASE}/v1/lora/scale",
            json={"scale": scale, "slot": slot},
            headers=API_HEADERS,
            timeout=30,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"  [!] Failed to set adapter scale: {e}")
        return False


def set_group_scales(self_attn: float, cross_attn: float, mlp: float, slot: int = 0) -> bool:
    """Set adapter group scales (SA/CA/MLP) via the API."""
    try:
        resp = requests.post(
            f"{API_BASE}/v1/lora/slot-group-scales",
            json={"slot": slot, "self_attn": self_attn, "cross_attn": cross_attn, "mlp": mlp},
            headers=API_HEADERS,
            timeout=30,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"  [!] Failed to set group scales: {e}")
        return False


def map_settings_to_request(settings: Dict[str, Any], thinking_override: Optional[bool] = None) -> Dict[str, Any]:
    """Map UI settings JSON to API GenerateMusicRequest fields."""
    # The settingstouse.json `prompt` field contains lyrics, `style` is the caption
    lyrics = settings.get("prompt", "")
    caption = settings.get("style", "")

    req = {
        "prompt": caption,
        "lyrics": lyrics,
        "thinking": thinking_override if thinking_override is not None else settings.get("thinking", True),
        "model": settings.get("model", ""),
        "bpm": settings.get("bpm"),
        "key_scale": settings.get("keyScale", ""),
        "time_signature": settings.get("timeSignature", ""),
        "vocal_language": settings.get("vocalLanguage", "en"),
        "audio_duration": settings.get("duration"),
        "inference_steps": settings.get("inferenceSteps", 60),
        "guidance_scale": settings.get("guidanceScale", 15),
        "batch_size": 1,
        "use_random_seed": False,
        "seed": settings.get("seed", 1),
        "audio_format": settings.get("audioFormat", "flac"),
        "infer_method": settings.get("inferMethod", "ode"),
        "scheduler": settings.get("scheduler", "linear"),
        "shift": settings.get("shift", 3.0),
        "lm_backend": settings.get("lmBackend", "vllm"),
        "lm_model_path": settings.get("lmModel", ""),
        "lm_temperature": settings.get("lmTemperature", 0.85),
        "lm_cfg_scale": settings.get("lmCfgScale", 2.5),
        "lm_top_k": settings.get("lmTopK"),
        "lm_top_p": settings.get("lmTopP", 0.9),
        "lm_negative_prompt": settings.get("lmNegativePrompt", ""),
        "lm_repetition_penalty": 1.0,
        "constrained_decoding": True,
        "allow_lm_batch": settings.get("allowLmBatch", True),
        "use_cot_caption": settings.get("useCotCaption", False),
        "use_cot_language": settings.get("useCotLanguage", False),
        "instruction": settings.get("instruction", ""),
        "guidance_mode": settings.get("guidanceMode", ""),
        "use_pag": settings.get("usePag", False),
        "pag_start": settings.get("pagStart", 0.3),
        "pag_end": settings.get("pagEnd", 0.7),
        "pag_scale": settings.get("pagScale", 0.25),
        "cfg_interval_start": settings.get("cfgIntervalStart", 0),
        "cfg_interval_end": settings.get("cfgIntervalEnd", 1),
        # Enable scoring and LRC
        "get_scores": True,
        "get_lrc": True,
        "score_scale": settings.get("scoreScale", 0.5),
        # Disable mastering/normalization for consistent comparison
        "auto_master": False,
        "enable_normalization": False,
        "task_type": "text2music",
    }
    return req


def run_generation(request_body: Dict[str, Any], timeout: int = 600) -> Optional[Dict]:
    """Submit a generation job and wait for completion.

    Returns the full API response dict, or None on failure.
    """
    try:
        resp = requests.post(
            f"{API_BASE}/v1/release_task",
            json=request_body,
            headers=API_HEADERS,
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Unwrap the response envelope
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return data
        else:
            print(f"  [!] Generation failed: HTTP {resp.status_code}")
            try:
                print(f"      {resp.json()}")
            except Exception:
                pass
            return None
    except requests.exceptions.Timeout:
        print("  [!] Generation timed out")
        return None
    except Exception as e:
        print(f"  [!] Generation error: {e}")
        return None


def check_api_health() -> bool:
    """Check if the API server is running."""
    try:
        resp = requests.get(f"{API_BASE}/v1/lora/status", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ── Result Extraction ─────────────────────────────────────────────────────


def extract_audio_path(result: Dict) -> Optional[str]:
    """Extract the audio file path from the generation response."""
    audios = result.get("audios", [])
    if audios and isinstance(audios[0], dict):
        url = audios[0].get("url", "")
        # URL format: /api/audio/filename.flac → need to resolve to filesystem path
        # Check if there's a direct path
        path = audios[0].get("path", "")
        if path and os.path.exists(path):
            return path
        # Try resolving from URL
        if url:
            # The API serves from temp_audio_dir
            basename = url.split("/")[-1] if "/" in url else url
            candidates = [
                os.path.join(PROJECT_ROOT, ".cache", "acestep", "tmp", "api_audio", basename),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
    return None


def extract_lrc_text(result: Dict) -> str:
    """Extract LRC text from the generation response."""
    audios = result.get("audios", [])
    if audios and isinstance(audios[0], dict):
        lrc = audios[0].get("lrc", "")
        if lrc:
            return lrc
    return ""


def extract_dit_score(result: Dict) -> float:
    """Extract DiT alignment score from the generation response."""
    scores = result.get("scores", {})
    dit_align = scores.get("dit_alignment", {})
    return float(dit_align.get("dit_score", 0.0))


def extract_pmi_score(result: Dict) -> float:
    """Extract PMI score from the generation response."""
    scores = result.get("scores", {})
    pmi = scores.get("pmi", {})
    return float(pmi.get("global", 0.0))


# ── Results IO ────────────────────────────────────────────────────────────

CSV_HEADERS = [
    "config_id",
    "label",
    "scale",
    "self_attn",
    "cross_attn",
    "mlp",
    "thinking",
    "combined_score",
    "whisper_score",
    "dit_score",
    "pmi_score",
    "lrc_score",
    "matched_lines",
    "missing_lines",
    "partial_lines",
    "total_lines",
    "lrc_healthy",
    "lrc_skipped",
    "whisper_preview",
    "audio_file",
    "gen_time_s",
]


def write_csv_row(csv_path: str, row: Dict[str, Any], write_header: bool = False):
    """Append a row to the results CSV."""
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_completed_configs(csv_path: str) -> set:
    """Load config IDs that have already been completed (for resume)."""
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row.get("config_id", ""))
    return completed


# ── Main Sweep ────────────────────────────────────────────────────────────


def run_sweep(
    settings_path: str,
    output_dir: Optional[str] = None,
    max_configs: Optional[int] = None,
    dry_run: bool = False,
    whisper_model: str = "small",
    whisper_device: str = "cuda",
):
    """Execute the full parameter sweep."""

    # ── Load settings ──
    print(f"\n{'='*70}")
    print("  ADAPTER ADHERENCE SWEEP TOOL")
    print(f"{'='*70}\n")

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    # Extract baseline params from settings
    adapter_slots = settings.get("adapterSlots", [])
    if adapter_slots:
        slot = adapter_slots[0]
        base_scale = slot.get("scale", 0.7)
        gs = slot.get("group_scales", {})
        base_sa = gs.get("self_attn", 1.0)
        base_ca = gs.get("cross_attn", 1.0)
        base_mlp = gs.get("mlp", 1.0)
    else:
        base_scale, base_sa, base_ca, base_mlp = 0.7, 0.35, 0.35, 1.8

    input_lyrics = settings.get("prompt", "")

    print(f"  Settings:    {settings_path}")
    print(f"  Adapter:     {settings.get('loraPath', 'N/A')}")
    print(f"  Baseline:    scale={base_scale} SA={base_sa} CA={base_ca} MLP={base_mlp}")
    print(f"  Seed:        {settings.get('seed', 'random')}")
    print(f"  Duration:    {settings.get('duration', '?')}s")
    print(f"  Inf Steps:   {settings.get('inferenceSteps', '?')}")
    print(f"  Whisper:     {whisper_model} on {whisper_device}")
    print(f"  Lyric lines: {len(extract_lyric_lines(input_lyrics))}")
    print()

    # ── Build configs ──
    configs = build_sweep_configs(base_scale, base_sa, base_ca, base_mlp)
    if max_configs:
        configs = configs[: max_configs]

    print(f"  Sweep configs: {len(configs)}")
    print(f"  {'─'*66}")

    for c in configs:
        print(f"  [{c.config_id}] {c.summary():50s} │ {c.label}")

    print(f"  {'─'*66}")

    if dry_run:
        print("\n  [DRY RUN] No generations will be performed.\n")
        return

    # ── Setup output directory ──
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_ROOT, "sweep_results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    diff_dir = os.path.join(output_dir, "diffs")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "results.csv")
    completed = load_completed_configs(csv_path)

    if completed:
        print(f"\n  Resuming: {len(completed)} configs already completed")

    # ── Check API is running ──
    print("\n  Checking API health...", end=" ")
    if not check_api_health():
        print("FAILED!")
        print("  [!] API server is not running at", API_BASE)
        print("  [!] Please start the app first via launch.bat")
        sys.exit(1)
    print("OK")

    # ── Initialise Whisper ──
    print(f"  Loading Whisper ({whisper_model})...", end=" ", flush=True)
    scorer = WhisperScorer(model_size=whisper_model, device=whisper_device)
    scorer._ensure_model()  # Pre-load to avoid delay on first generation
    print("OK")

    # ── Write CSV header if new ──
    if not completed:
        write_csv_row(csv_path, {}, write_header=True)

    # ── Run sweep ──
    results: List[Dict] = []
    total = len(configs)
    start_time = time.time()

    for i, cfg in enumerate(configs):
        if cfg.config_id in completed:
            print(f"\n  [{i+1}/{total}] {cfg.config_id} — SKIPPED (already completed)")
            continue

        print(f"\n  {'═'*66}")
        print(f"  [{i+1}/{total}] Config {cfg.config_id}: {cfg.label}")
        print(f"  {cfg.summary()}")
        print(f"  {'─'*66}")

        # 1. Set adapter scales
        print("  Setting adapter scale...", end=" ", flush=True)
        set_adapter_scale(cfg.adapter_scale, slot=0)
        print("OK")

        print("  Setting group scales...", end=" ", flush=True)
        set_group_scales(cfg.self_attn, cfg.cross_attn, cfg.mlp, slot=0)
        print("OK")

        # Small delay to let the adapter apply
        time.sleep(0.5)

        # 2. Run generation
        print("  Generating audio...", end=" ", flush=True)
        gen_start = time.time()
        request_body = map_settings_to_request(settings, thinking_override=cfg.thinking)
        gen_result = run_generation(request_body)
        gen_time = round(time.time() - gen_start, 1)

        if gen_result is None:
            print(f"FAILED ({gen_time}s)")
            row = {k: "" for k in CSV_HEADERS}
            row.update(
                {
                    "config_id": cfg.config_id,
                    "label": cfg.label,
                    "scale": cfg.adapter_scale,
                    "self_attn": cfg.self_attn,
                    "cross_attn": cfg.cross_attn,
                    "mlp": cfg.mlp,
                    "thinking": cfg.thinking,
                    "gen_time_s": gen_time,
                    "whisper_preview": "GENERATION FAILED",
                }
            )
            write_csv_row(csv_path, row)
            results.append(row)
            continue

        print(f"OK ({gen_time}s)")

        # 3. Extract results
        audio_path = extract_audio_path(gen_result)
        lrc_text = extract_lrc_text(gen_result)
        dit_score = extract_dit_score(gen_result)
        pmi_score = extract_pmi_score(gen_result)

        # Copy audio to sweep output
        saved_audio = ""
        if audio_path and os.path.exists(audio_path):
            import shutil

            ext = os.path.splitext(audio_path)[1]
            saved_name = f"{cfg.config_id}{ext}"
            saved_path = os.path.join(audio_dir, saved_name)
            shutil.copy2(audio_path, saved_path)
            saved_audio = saved_name
            print(f"  Audio saved: {saved_name}")

        # Save LRC
        if lrc_text:
            lrc_path = os.path.join(diff_dir, f"{cfg.config_id}.lrc")
            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_text)

        # 4. Whisper transcription
        whisper_result = None
        if audio_path and os.path.exists(audio_path):
            print("  Transcribing with Whisper...", end=" ", flush=True)
            try:
                whisper_result = scorer.score(audio_path, input_lyrics, language="en")
                print(f"OK (similarity={whisper_result.overall_similarity:.3f})")
                print(
                    f"  Lines: {whisper_result.matched_count}✅ "
                    f"{whisper_result.partial_count}⚠️ "
                    f"{whisper_result.missing_count}❌ "
                    f"/ {whisper_result.total_input_lines}"
                )
            except Exception as e:
                print(f"FAILED: {e}")

        # 5. LRC scoring
        lrc_result = None
        if lrc_text:
            lrc_result = score_lrc_timing(lrc_text, total_duration=settings.get("duration", 0))
            print(
                f"  LRC: {lrc_result.healthy_lines} healthy, "
                f"{lrc_result.skipped_lines} skipped "
                f"(score={lrc_result.score:.3f})"
            )

        # 6. Combined score
        adherence = compute_adherence_score(
            whisper_result=whisper_result,
            lrc_result=lrc_result,
            dit_score=dit_score,
            pmi_score=pmi_score,
        )

        print(f"  ──────────────────")
        print(f"  COMBINED: {adherence.combined_score:.4f}")
        print(f"    Whisper={adherence.whisper_score:.3f}  DiT={adherence.dit_score:.3f}  LRC={adherence.lrc_score:.3f}  PMI={adherence.pmi_score:.3f}")

        # 7. Save diff file
        if whisper_result:
            diff_path = os.path.join(diff_dir, f"{cfg.config_id}_diff.txt")
            with open(diff_path, "w", encoding="utf-8") as f:
                f.write(f"Config: {cfg.config_id} — {cfg.label}\n")
                f.write(f"Params: {cfg.summary()}\n")
                f.write(f"Overall Similarity: {whisper_result.overall_similarity:.4f}\n")
                f.write(f"{'='*70}\n\n")
                f.write(f"RAW TRANSCRIPTION:\n{whisper_result.raw_transcription}\n\n")
                f.write(f"{'='*70}\n")
                f.write(f"PER-LINE ANALYSIS:\n{'─'*70}\n")
                for ld in whisper_result.line_diffs:
                    icon = {"matched": "✅", "partial": "⚠️", "missing": "❌"}.get(ld.status, "?")
                    f.write(f"\n{icon} [{ld.status}] (sim={ld.similarity:.2f})\n")
                    f.write(f"  INPUT:   {ld.input_line}\n")
                    if ld.matched_text:
                        f.write(f"  HEARD:   {ld.matched_text}\n")

        # 8. Write CSV row
        preview = ""
        if whisper_result:
            preview = whisper_result.raw_transcription[:100].replace("\n", " ")

        row = {
            "config_id": cfg.config_id,
            "label": cfg.label,
            "scale": cfg.adapter_scale,
            "self_attn": cfg.self_attn,
            "cross_attn": cfg.cross_attn,
            "mlp": cfg.mlp,
            "thinking": cfg.thinking,
            "combined_score": adherence.combined_score,
            "whisper_score": adherence.whisper_score,
            "dit_score": adherence.dit_score,
            "pmi_score": adherence.pmi_score,
            "lrc_score": adherence.lrc_score,
            "matched_lines": whisper_result.matched_count if whisper_result else "",
            "missing_lines": whisper_result.missing_count if whisper_result else "",
            "partial_lines": whisper_result.partial_count if whisper_result else "",
            "total_lines": whisper_result.total_input_lines if whisper_result else "",
            "lrc_healthy": lrc_result.healthy_lines if lrc_result else "",
            "lrc_skipped": lrc_result.skipped_lines if lrc_result else "",
            "whisper_preview": preview,
            "audio_file": saved_audio,
            "gen_time_s": gen_time,
        }
        write_csv_row(csv_path, row)
        results.append(row)

    # ── Final Summary ─────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    scorer.unload()

    print(f"\n\n{'═'*70}")
    print(f"  SWEEP COMPLETE — {len(results)} configs tested in {elapsed}s")
    print(f"{'═'*70}\n")

    # Sort by combined score
    scored_results = [r for r in results if r.get("combined_score") not in ("", None)]
    scored_results.sort(key=lambda r: float(r.get("combined_score", 0)), reverse=True)

    if scored_results:
        print(f"  {'Rank':<5} {'ID':>4} {'Score':>6} {'Whis':>6} {'DiT':>6} {'LRC':>6} {'Miss':>5} {'Label'}")
        print(f"  {'─'*65}")
        for rank, r in enumerate(scored_results[:10], 1):
            marker = " 🏆" if rank <= 3 else ""
            print(
                f"  {rank:<5} {r['config_id']:>4} "
                f"{float(r.get('combined_score',0)):>6.3f} "
                f"{float(r.get('whisper_score',0)):>6.3f} "
                f"{float(r.get('dit_score',0)):>6.3f} "
                f"{float(r.get('lrc_score',0)):>6.3f} "
                f"{r.get('missing_lines','?'):>5} "
                f"{r['label']}{marker}"
            )

    # Write summary file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"ADAPTER ADHERENCE SWEEP RESULTS\n")
        f.write(f"{'='*70}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Settings: {settings_path}\n")
        f.write(f"Adapter: {settings.get('loraPath', 'N/A')}\n")
        f.write(f"Total configs: {len(results)}\n")
        f.write(f"Total time: {elapsed}s\n\n")
        f.write(f"RANKING (by combined adherence score):\n")
        f.write(f"{'─'*70}\n")
        for rank, r in enumerate(scored_results, 1):
            f.write(
                f"#{rank:>2}  [{r['config_id']}] {r['label']}\n"
                f"     Combined={float(r.get('combined_score',0)):.4f}  "
                f"Whisper={float(r.get('whisper_score',0)):.4f}  "
                f"DiT={float(r.get('dit_score',0)):.4f}  "
                f"LRC={float(r.get('lrc_score',0)):.4f}  "
                f"PMI={float(r.get('pmi_score',0)):.4f}\n"
                f"     scale={r['scale']} SA={r['self_attn']} CA={r['cross_attn']} "
                f"MLP={r['mlp']} thinking={r['thinking']}\n"
                f"     Missing lines: {r.get('missing_lines','?')} / {r.get('total_lines','?')}\n\n"
            )

    print(f"\n  Results saved to: {output_dir}")
    print(f"    CSV:     {csv_path}")
    print(f"    Summary: {summary_path}")
    print(f"    Audio:   {audio_dir}")
    print(f"    Diffs:   {diff_dir}")
    print()


# ── CLI Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapter Adherence Sweep Tool")
    parser.add_argument("--config", required=True, help="Path to settingstouse.json")
    parser.add_argument("--dry-run", action="store_true", help="Print config matrix without running")
    parser.add_argument("--max-configs", type=int, default=None, help="Max configs to test")
    parser.add_argument("--whisper-model", default="small", choices=["tiny", "small", "medium", "large-v3"], help="Whisper model size")
    parser.add_argument("--whisper-device", default="cuda", choices=["cuda", "cpu"], help="Whisper device")
    parser.add_argument("--output-dir", default=None, help="Custom output directory")
    parser.add_argument("--api-url", default="http://localhost:8001", help="API base URL")

    args = parser.parse_args()

    API_BASE = args.api_url

    run_sweep(
        settings_path=args.config,
        output_dir=args.output_dir,
        max_configs=args.max_configs,
        dry_run=args.dry_run,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
    )
