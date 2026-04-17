"""SuperSep — Multi-Stage Stem Separation Pipeline (Library Module).

Chains the best open-source AI models to extract up to 17 high-quality
stems from any audio file, fully CUDA-accelerated.

Pipeline:
  Stage 1 (BS-RoFormer SW):      Vocals, Bass, Drums, Guitar, Piano, Other
  Stage 2 (Mel-Band Karaoke):    Lead Vocals, Backing Vocals
  Stage 3 (MDX23C DrumSep):      Kick, Snare, Toms, Hi-Hat, Ride, Crash
  Stage 4 (Demucs htdemucs_6s):  Refine "Other" -> guitar, piano, vocals, bass, drums, residual

Adapted from D:\\Ace-Step-Latest\\SuperSep\\supersep.py for integration
into HOT-Step 9000.
"""

from __future__ import annotations

import os
import re
import shutil
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional
from uuid import uuid4

import numpy as np
import soundfile as sf
from loguru import logger


# ── Model Configuration ─────────────────────────────────────────────────

STAGE1_MODEL = "BS-Roformer-SW.ckpt"
STAGE2_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
STAGE3_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
STAGE4_MODEL = "htdemucs_6s.yaml"
FALLBACK_MODEL = "htdemucs_ft.yaml"
MODEL_SAMPLE_RATE = 44100

# Separation levels (which stages to run)
SEPARATION_LEVELS = {
    "basic": [1],           # 6 stems
    "vocal-split": [1, 2],  # 8 stems (vocals split into lead/backing)
    "full": [1, 2, 3],      # 14 stems (+ 6 drum stems)
    "maximum": [1, 2, 3, 4],  # up to 17 stems
}


# ── Data Model ───────────────────────────────────────────────────────────

class SuperSepStem:
    """Result descriptor for a separated stem."""

    __slots__ = ("id", "stem_type", "stem_name", "file_path", "file_name",
                 "category", "stage", "duration")

    def __init__(
        self,
        stem_type: str,
        stem_name: str,
        file_path: str,
        file_name: str,
        category: str = "other",
        stage: int = 1,
        duration: float = 0.0,
        id: Optional[str] = None,
    ):
        self.id = id or str(uuid4())
        self.stem_type = stem_type
        self.stem_name = stem_name
        self.file_path = file_path
        self.file_name = file_name
        self.category = category
        self.stage = stage
        self.duration = duration

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "stem_type": self.stem_type,
            "stem_name": self.stem_name,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "category": self.category,
            "stage": self.stage,
            "duration": self.duration,
        }


# ── BFloat16 workaround ─────────────────────────────────────────────────

@contextmanager
def _float32_default_dtype():
    """Temporarily force torch default dtype to float32.

    ACE-Step sets default dtype to bfloat16 for GPU inference.  Models from
    audio-separator may create tensors that inherit this dtype, causing MKL
    FFT crashes on Windows.
    """
    import torch
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_audio_duration(path: str) -> float:
    """Return duration in seconds (best-effort)."""
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 0.0


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return "".join(c if c.isalnum() or c in " -_.()[]" else "_" for c in name).strip()


def _find_output(output_dir: str, keyword: str) -> Optional[str]:
    """Find the first file in output_dir whose name contains keyword."""
    d = Path(output_dir)
    if not d.exists():
        return None
    for f in sorted(d.iterdir()):
        if keyword.lower() in f.name.lower() and f.is_file():
            return str(f)
    return None


def _find_all_outputs(output_dir: str) -> List[str]:
    """Return all audio files in a directory."""
    d = Path(output_dir)
    if not d.exists():
        return []
    exts = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
    return [str(f) for f in sorted(d.iterdir()) if f.suffix.lower() in exts and f.is_file()]


def _is_silent(file_path: str, threshold_db: float = -60.0) -> bool:
    """Check if an audio file is effectively silent."""
    try:
        data, sr = sf.read(file_path, dtype='float32')
        peak = np.max(np.abs(data))
        if peak == 0:
            return True
        peak_db = 20 * np.log10(peak)
        return peak_db < threshold_db
    except Exception:
        return False


def _get_separator(output_dir: str, model_file_dir: str = None):
    """Create a fresh Separator instance."""
    from audio_separator.separator import Separator
    return Separator(
        output_dir=output_dir,
        model_file_dir=model_file_dir,
        output_format="FLAC",
        normalization_threshold=0.9,
        output_single_stem=None,
        sample_rate=MODEL_SAMPLE_RATE,
    )


def _get_default_model_dir() -> str:
    """Resolve default model directory."""
    # Check env var first
    env_dir = os.environ.get("SUPERSEP_MODEL_DIR")
    if env_dir and Path(env_dir).is_dir():
        return env_dir
    # Default: models/supersep/ in project root
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / "models" / "supersep")


ProgressCallback = Callable[[int, str, float], None]
"""Signature: (stage_number, message, progress_0_to_1)"""


# ── Main Pipeline ────────────────────────────────────────────────────────

def run_supersep(
    audio_path: str,
    output_dir: str,
    *,
    model_dir: Optional[str] = None,
    level: str = "full",
    progress_callback: Optional[ProgressCallback] = None,
) -> List[SuperSepStem]:
    """Run the SuperSep pipeline on an audio file.

    Args:
        audio_path: Path to the source audio file.
        output_dir: Directory to write output stems.
        model_dir: Directory containing model files.
        level: Separation level — "basic", "vocal-split", "full", or "maximum".
        progress_callback: Called with (stage, message, percent).

    Returns:
        List of SuperSepStem objects describing each output stem.
    """
    if model_dir is None:
        model_dir = _get_default_model_dir()

    stages = SEPARATION_LEVELS.get(level, SEPARATION_LEVELS["full"])
    audio_path = str(Path(audio_path).resolve())

    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tmp_base = Path(output_dir) / "_tmp"
    tmp_base.mkdir(exist_ok=True)

    cb = progress_callback or (lambda *_: None)
    final_stems: Dict[str, dict] = {}
    total_start = time.time()

    # Track stage stems to pass between stages
    s1_vocals = s1_bass = s1_drums = s1_guitar = s1_piano = s1_other = None

    # ── STAGE 1: Primary 6-stem split ────────────────────────────────
    if 1 in stages:
        cb(1, "Loading BS-RoFormer SW model...", 0.05)
        logger.info("[SuperSep] Stage 1: Primary 6-stem split")

        tmp_s1 = str(tmp_base / "stage1")
        os.makedirs(tmp_s1, exist_ok=True)

        try:
            with _float32_default_dtype():
                sep = _get_separator(output_dir=tmp_s1, model_file_dir=model_dir)
                sep.load_model(model_filename=STAGE1_MODEL)
                cb(1, "Separating into 6 primary stems...", 0.15)
                sep.separate(audio_path)

            s1_vocals = _find_output(tmp_s1, "vocals")
            s1_bass = _find_output(tmp_s1, "bass")
            s1_drums = _find_output(tmp_s1, "drums")
            s1_guitar = _find_output(tmp_s1, "guitar")
            s1_piano = _find_output(tmp_s1, "piano")
            s1_other = _find_output(tmp_s1, "other")

            found = sum(1 for x in [s1_vocals, s1_bass, s1_drums, s1_guitar, s1_piano, s1_other] if x)
            logger.info(f"[SuperSep] Stage 1: {found} stems extracted")
            cb(1, f"Stage 1 complete — {found} stems", 0.25)

        except Exception as e:
            logger.error(f"[SuperSep] Stage 1 failed: {e}")
            cb(1, f"Stage 1 failed, trying fallback...", 0.15)
            try:
                with _float32_default_dtype():
                    sep = _get_separator(output_dir=tmp_s1, model_file_dir=model_dir)
                    sep.load_model(model_filename=FALLBACK_MODEL)
                    sep.separate(audio_path)
                s1_vocals = _find_output(tmp_s1, "vocals")
                s1_bass = _find_output(tmp_s1, "bass")
                s1_drums = _find_output(tmp_s1, "drums")
                s1_other = _find_output(tmp_s1, "other")
                cb(1, "Fallback: 4 stems extracted", 0.25)
            except Exception as e2:
                logger.error(f"[SuperSep] Fallback also failed: {e2}")
                raise RuntimeError(f"Stage 1 failed: {e}; Fallback: {e2}")

        # Keep direct stems (not further processed)
        if s1_bass:
            final_stems["01_Bass"] = {"path": s1_bass, "category": "instruments", "name": "Bass"}
        if s1_guitar:
            final_stems["02_Guitar"] = {"path": s1_guitar, "category": "instruments", "name": "Guitar"}
        if s1_piano:
            final_stems["03_Piano"] = {"path": s1_piano, "category": "instruments", "name": "Piano"}

        # If not splitting vocals further, keep as-is
        if 2 not in stages and s1_vocals:
            final_stems["04_Vocals"] = {"path": s1_vocals, "category": "vocals", "name": "Vocals"}
        # If not splitting drums further, keep as-is
        if 3 not in stages and s1_drums:
            final_stems["05_Drums"] = {"path": s1_drums, "category": "drums", "name": "Drums"}
        # If not refining other, keep as-is
        if 4 not in stages and s1_other:
            final_stems["06_Other"] = {"path": s1_other, "category": "other", "name": "Other"}

    # ── STAGE 2: Vocal sub-separation ────────────────────────────────
    if 2 in stages and s1_vocals:
        cb(2, "Loading Mel-Band RoFormer Karaoke model...", 0.30)
        logger.info("[SuperSep] Stage 2: Vocal sub-separation")

        tmp_s2 = str(tmp_base / "stage2")
        os.makedirs(tmp_s2, exist_ok=True)

        try:
            with _float32_default_dtype():
                sep = _get_separator(output_dir=tmp_s2, model_file_dir=model_dir)
                sep.load_model(model_filename=STAGE2_MODEL)
                cb(2, "Splitting lead / backing vocals...", 0.35)
                sep.separate(s1_vocals)

            # Parse output filenames to find lead vs backing
            s2_lead = s2_backing = None
            for f in sorted(Path(tmp_s2).iterdir()):
                if not f.is_file():
                    continue
                labels = re.findall(r'\(([^)]+)\)', f.stem)
                if not labels:
                    continue
                last_label = labels[-1].lower()
                if last_label == "vocals":
                    s2_lead = str(f)
                elif last_label == "instrumental":
                    s2_backing = str(f)

            if not s2_lead and not s2_backing:
                s2_files = _find_all_outputs(tmp_s2)
                if len(s2_files) >= 2:
                    s2_lead, s2_backing = s2_files[0], s2_files[1]
                elif len(s2_files) == 1:
                    s2_lead = s2_files[0]

            if s2_lead:
                final_stems["04_Lead_Vocals"] = {"path": s2_lead, "category": "vocals", "name": "Lead Vocals"}
            if s2_backing:
                final_stems["05_Backing_Vocals"] = {"path": s2_backing, "category": "vocals", "name": "Backing Vocals"}
            if not s2_lead and not s2_backing:
                final_stems["04_Vocals"] = {"path": s1_vocals, "category": "vocals", "name": "Vocals"}

            cb(2, "Vocal split complete", 0.45)

        except Exception as e:
            logger.warning(f"[SuperSep] Stage 2 failed: {e}")
            final_stems["04_Vocals"] = {"path": s1_vocals, "category": "vocals", "name": "Vocals"}
            cb(2, "Vocal split failed, keeping original", 0.45)

    # ── STAGE 3: Drum sub-separation ─────────────────────────────────
    if 3 in stages and s1_drums:
        cb(3, "Loading MDX23C DrumSep model...", 0.50)
        logger.info("[SuperSep] Stage 3: Drum sub-separation")

        tmp_s3 = str(tmp_base / "stage3")
        os.makedirs(tmp_s3, exist_ok=True)

        try:
            with _float32_default_dtype():
                sep = _get_separator(output_dir=tmp_s3, model_file_dir=model_dir)
                sep.load_model(model_filename=STAGE3_MODEL)
                cb(3, "Splitting drums into 6 components...", 0.55)
                sep.separate(s1_drums)

            drum_map = {
                "kick": ("06_Kick", "Kick"),
                "snare": ("07_Snare", "Snare"),
                "toms": ("08_Toms", "Toms"),
                "hh": ("09_HiHat", "Hi-Hat"),
                "ride": ("10_Ride", "Ride"),
                "crash": ("11_Crash", "Crash"),
            }

            drum_count = 0
            for keyword, (key, name) in drum_map.items():
                found = _find_output(tmp_s3, keyword)
                if found:
                    final_stems[key] = {"path": found, "category": "drums", "name": name}
                    drum_count += 1

            if drum_count == 0:
                final_stems["06_Drums"] = {"path": s1_drums, "category": "drums", "name": "Drums"}

            cb(3, f"Drum split complete — {drum_count} components", 0.70)

        except Exception as e:
            logger.warning(f"[SuperSep] Stage 3 failed: {e}")
            final_stems["06_Drums"] = {"path": s1_drums, "category": "drums", "name": "Drums"}
            cb(3, "Drum split failed, keeping original", 0.70)

    # ── STAGE 4: "Other" refinement ──────────────────────────────────
    if 4 in stages and s1_other:
        cb(4, "Loading Demucs htdemucs_6s model...", 0.75)
        logger.info("[SuperSep] Stage 4: 'Other' refinement")

        tmp_s4 = str(tmp_base / "stage4")
        os.makedirs(tmp_s4, exist_ok=True)

        try:
            with _float32_default_dtype():
                sep = _get_separator(output_dir=tmp_s4, model_file_dir=model_dir)
                sep.load_model(model_filename=STAGE4_MODEL)
                cb(4, "Refining 'other' stem...", 0.80)
                sep.separate(s1_other)

            other_map = {
                "vocals": ("12_Other_Vocal_Bleed", "Vocal Bleed", "other"),
                "guitar": ("13_Other_Guitar", "Other Guitar", "instruments"),
                "piano": ("14_Other_Piano_Keys", "Other Piano/Keys", "instruments"),
                "bass": ("15_Other_Bass", "Other Bass", "instruments"),
                "drums": ("16_Other_Percussion", "Other Percussion", "drums"),
                "other": ("17_Residual", "Residual (Synths/FX)", "other"),
            }

            for keyword, (key, name, cat) in other_map.items():
                found = _find_output(tmp_s4, keyword)
                if found:
                    final_stems[key] = {"path": found, "category": cat, "name": name}

            cb(4, "Other refinement complete", 0.90)

        except Exception as e:
            logger.warning(f"[SuperSep] Stage 4 failed: {e}")
            final_stems["12_Other"] = {"path": s1_other, "category": "other", "name": "Other"}
            cb(4, "Other refinement failed, keeping original", 0.90)

    # ── Collect final stems ──────────────────────────────────────────
    cb(0, "Collecting final stems...", 0.92)

    result: List[SuperSepStem] = []
    for stem_key, info in sorted(final_stems.items()):
        src = Path(info["path"])
        if _is_silent(str(src)):
            logger.debug(f"[SuperSep] Skipping silent stem: {stem_key}")
            continue
        # Copy to output dir with clean name
        dst = Path(output_dir) / f"{stem_key}{src.suffix}"
        shutil.copy2(str(src), str(dst))
        result.append(SuperSepStem(
            stem_type=stem_key,
            stem_name=info["name"],
            file_path=str(dst),
            file_name=dst.name,
            category=info.get("category", "other"),
            stage=int(stem_key.split("_")[0]) if stem_key[0].isdigit() else 0,
            duration=_get_audio_duration(str(dst)),
        ))

    # Cleanup temp
    try:
        shutil.rmtree(str(tmp_base))
    except Exception:
        pass

    elapsed = time.time() - total_start
    logger.info(f"[SuperSep] Done — {len(result)} stems in {elapsed:.1f}s")
    cb(0, f"Complete — {len(result)} stems extracted", 1.0)

    return result


# ── Stem Recombination ───────────────────────────────────────────────────

def recombine_stems(
    stems: List[dict],
    output_path: str,
    *,
    sample_rate: int = MODEL_SAMPLE_RATE,
) -> str:
    """Mix stems back together with per-stem volume and mute controls.

    Args:
        stems: List of dicts with keys: path, volume (0.0-2.0), muted (bool)
        output_path: Where to save the mixed audio file.
        sample_rate: Target sample rate.

    Returns:
        Absolute path to the mixed file.
    """
    mixed = None
    max_len = 0

    # First pass: determine max length
    for stem in stems:
        if stem.get("muted", False):
            continue
        vol = float(stem.get("volume", 1.0))
        if vol <= 0.0:
            continue
        data, sr = sf.read(stem["path"], dtype="float32")
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)  # mono → stereo
        max_len = max(max_len, len(data))

    if max_len == 0:
        raise ValueError("All stems are muted or empty — nothing to mix")

    # Second pass: mix
    mixed = np.zeros((max_len, 2), dtype=np.float64)

    for stem in stems:
        if stem.get("muted", False):
            continue
        vol = float(stem.get("volume", 1.0))
        if vol <= 0.0:
            continue
        data, sr = sf.read(stem["path"], dtype="float32")
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        # Pad if shorter
        if len(data) < max_len:
            pad = np.zeros((max_len - len(data), data.shape[1]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

        mixed += data.astype(np.float64) * vol

    # Normalize to prevent clipping (-1dB headroom)
    peak = np.max(np.abs(mixed))
    if peak > 0:
        target = 10 ** (-1.0 / 20)  # -1dB
        mixed = mixed * (target / peak)

    mixed = mixed.astype(np.float32)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mixed, sample_rate, format="FLAC")
    logger.info(f"[SuperSep] Recombined {len(stems)} stems → {output_path}")

    return str(Path(output_path).resolve())
