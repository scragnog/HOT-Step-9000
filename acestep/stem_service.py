"""Server-side stem separation service using audio_separator.

Models (lazy-downloaded on first use):
  - BS-RoFormer         for vocals/instrumental  (SDR 12.97)
  - BS-Roformer-SW      for 6-stem separation    (vocals, drums, bass, guitar, piano, other)

Available modes:
  - ``vocals``     : BS-RoFormer → 2 stems (vocals + instrumental)
  - ``every-stem`` : BS-Roformer-SW → 6 stems (vocals, drums, bass, guitar, piano, other)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from loguru import logger


# ---------------------------------------------------------------------------
# BFloat16 workaround: ACE-Step sets torch default dtype to bfloat16
# for GPU inference.  BSRoformer.__init__ calls torch.stft(torch.randn(...))
# which inherits that dtype, and MKL FFT can't handle bfloat16 on Windows.
# Fix: temporarily restore float32 default during model load.
# ---------------------------------------------------------------------------

@contextmanager
def _float32_default_dtype():
    """Temporarily force torch default dtype to float32.

    ACE-Step's init_service_orchestrator sets the default dtype to bfloat16.
    BSRoformer's constructor calls torch.stft(torch.randn(1, 4096), ...)
    which creates a bfloat16 tensor — MKL FFT crashes on it (Windows only,
    fixed in PyTorch >= 2.10).
    """
    import torch
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class StemInfo:
    """Lightweight stem result descriptor."""

    __slots__ = ("id", "stem_type", "file_path", "file_name", "duration")

    def __init__(
        self,
        stem_type: str,
        file_path: str,
        file_name: str,
        duration: float = 0.0,
        id: Optional[str] = None,
    ):
        self.id = id or str(uuid4())
        self.stem_type = stem_type
        self.file_path = file_path
        self.file_name = file_name
        self.duration = duration

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "stem_type": self.stem_type,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "duration": self.duration,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_audio_duration(path: str) -> float:
    """Return duration in seconds (best-effort, returns 0.0 on failure)."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        pass
    try:
        import librosa
        dur = librosa.get_duration(path=path)
        return float(dur)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class StemService:
    """Singleton service for audio stem separation.

    Thread-safe lazy initialisation.  The underlying ``Separator``
    instance is created on first call and reused thereafter.
    """

    ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    ROFORMER_SW_MODEL = "BS-Roformer-SW.ckpt"

    def __init__(self, output_root: Optional[str] = None, device: str = "auto"):
        self._separator = None
        self._lock = threading.Lock()
        self._device = device
        # Default output root next to project dir
        if output_root is None:
            project_root = Path(__file__).resolve().parent.parent
            output_root = str(project_root / "stems_output")
        self._output_root = output_root
        os.makedirs(self._output_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _get_separator(self):
        """Lazy-init audio-separator."""
        if self._separator is None:
            with self._lock:
                if self._separator is None:
                    logger.info("[StemService] Loading audio-separator…")
                    from audio_separator.separator import Separator
                    self._separator = Separator()
                    logger.info("[StemService] audio-separator ready")
        return self._separator

    def is_available(self) -> bool:
        """Check whether audio_separator can be imported."""
        try:
            from audio_separator.separator import Separator  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def separate(
        self,
        audio_path: str,
        mode: str = "every-stem",
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[StemInfo]:
        """Separate *audio_path* into stems.

        Modes
        -----
        - ``vocals``     : BS-RoFormer → 2 stems (vocals + instrumental)
        - ``every-stem`` : BS-Roformer-SW → 6 stems (vocals, drums, bass, guitar, piano, other)
        """
        job_id = str(uuid4())
        output_dir = Path(self._output_root) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        separator = self._get_separator()

        dispatch = {
            "vocals": self._separate_vocals,
            "every-stem": self._separate_every_stem,
        }
        handler = dispatch.get(mode)
        if handler is None:
            raise ValueError(f"Unknown stem mode: {mode!r}. "
                             f"Choose from: {', '.join(dispatch)}")

        # Wrap the ENTIRE separation pipeline in float32 default dtype.
        # ACE-Step sets default dtype to bfloat16 for GPU inference, but
        # audio_separator's models (both RoFormer and Demucs) create
        # internal tensors that inherit the default dtype — crash or
        # dtype-mismatch on Windows MKL FFT and convolution layers.
        with _float32_default_dtype():
            return handler(separator, audio_path, output_dir, progress_callback)

    # ------------------------------------------------------------------
    # Internal separation strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(fpath: str, output_dir: Path) -> Path:
        p = Path(fpath)
        return p if p.is_absolute() else output_dir / p

    @staticmethod
    def _classify_stem_type(fname_lower: str, candidates: tuple) -> str:
        for c in candidates:
            if c in fname_lower:
                return c
        return "other"

    # ---- vocals only (RoFormer) ----

    def _separate_vocals(self, sep, audio_path, output_dir, cb) -> List[StemInfo]:
        if cb:
            cb("Loading BS-RoFormer model…", 0.1)

        sep.output_dir = str(output_dir)
        sep.output_format = "flac"
        sep.load_model(model_filename=self.ROFORMER_MODEL)

        if cb:
            cb("Separating vocals…", 0.3)

        files = sep.separate(audio_path)

        stems: List[StemInfo] = []
        for fp in files:
            fp = self._resolve(str(fp), output_dir)
            stem_type = "vocals" if "vocal" in fp.stem.lower() else "instrumental"
            stems.append(StemInfo(
                stem_type=stem_type,
                file_path=str(fp),
                file_name=fp.name,
                duration=_get_audio_duration(str(fp)),
            ))

        if cb:
            cb("Vocal separation complete", 1.0)
        return stems

    # ---- every-stem (BS-Roformer-SW 6-stem) ----

    def _separate_every_stem(self, sep, audio_path, output_dir, cb) -> List[StemInfo]:
        if cb:
            cb("Loading BS-Roformer-SW model…", 0.1)

        sep.output_dir = str(output_dir)
        sep.output_format = "flac"
        sep.load_model(model_filename=self.ROFORMER_SW_MODEL)

        if cb:
            cb("Separating all 6 stems…", 0.3)

        files = sep.separate(audio_path)

        stems: List[StemInfo] = []
        for fp in files:
            fp = self._resolve(str(fp), output_dir)
            stem_type = self._classify_stem_type(
                fp.stem.lower(),
                ("vocals", "drums", "bass", "guitar", "piano", "other"),
            )
            stems.append(StemInfo(
                stem_type=stem_type,
                file_path=str(fp),
                file_name=fp.name,
                duration=_get_audio_duration(str(fp)),
            ))
        if cb:
            cb("6-stem separation complete", 1.0)
        return stems

