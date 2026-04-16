"""Multi-backend vocoder service for audio post-processing.

Supports:
  - ADaMoSHiFiGANV1 (existing, mel-roundtrip via ConvNeXt + HiFi-GAN)
  - BigVGAN v2 (Snake activation, anti-aliased, full-band 44kHz)
  - Vocos (iSTFT-based, spectral-domain reconstruction)

Each backend is auto-detected from its checkpoint directory contents.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torchaudio
from loguru import logger

VOCODER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "checkpoints",
)

# ---------------------------------------------------------------------------
# Vocoder type detection
# ---------------------------------------------------------------------------

# Known vocoder signatures in config.json.  If a checkpoint folder matches
# one of these, it is treated as a vocoder; everything else is ignored.
_VOCODER_SIGNATURES = {
    "adamos": lambda cfg: cfg.get("_class_name") == "ADaMoSHiFiGANV1",
    "bigvgan": lambda cfg: (
        "bigvgan" in cfg.get("model_name", "").lower()
        or cfg.get("_name_or_path", "").startswith("nvidia/bigvgan")
        or "amp_block" in str(cfg.get("resblock", ""))
        or cfg.get("activation", "") == "snakebeta"
    ),
    "vocos": lambda cfg: cfg.get("_class_name", "").lower().startswith("vocos")
        or "vocos" in cfg.get("model_type", "").lower(),
}


def _detect_vocoder_type(checkpoint_dir: str) -> Optional[str]:
    """Return the vocoder backend type for a checkpoint directory, or None."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Check each known signature
    for vtype, matcher in _VOCODER_SIGNATURES.items():
        try:
            if matcher(cfg):
                return vtype
        except Exception:
            continue

    # Fallback: check for BigVGAN generator file
    if os.path.isfile(os.path.join(checkpoint_dir, "bigvgan_generator.pt")):
        return "bigvgan"

    # Fallback: check for Vocos-specific files
    if os.path.isfile(os.path.join(checkpoint_dir, "pytorch_model.bin")):
        # Re-check config for vocos markers
        if "backbone" in cfg and "head" in cfg:
            return "vocos"

    return None


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class VocoderBackend(ABC):
    """Base class for vocoder backends."""

    backend_type: str = "unknown"
    sample_rate: int = 44100

    @abstractmethod
    def load(self, checkpoint_dir: str, device: str) -> None:
        """Load model weights from *checkpoint_dir*."""

    @abstractmethod
    def enhance(self, waveform: torch.Tensor, input_sr: int) -> torch.Tensor:
        """Enhance *waveform* ``[C, T]`` and return ``[C, T']``."""

    def to(self, device: str) -> "VocoderBackend":
        """Move model to *device*."""
        return self

    def unload(self) -> None:
        """Free model weights."""
        pass


# ---------------------------------------------------------------------------
# ADaMoS backend (existing)
# ---------------------------------------------------------------------------

class ADaMoSBackend(VocoderBackend):
    """Wraps the existing ADaMoSHiFiGANV1 model."""

    backend_type = "adamos"
    sample_rate = 44100

    def __init__(self):
        self.model = None
        self.device = "cpu"

    def load(self, checkpoint_dir: str, device: str) -> None:
        from acestep.core.audio.music_vocoder import ADaMoSHiFiGANV1
        logger.info(f"[ADaMoS] Loading from {checkpoint_dir}")
        self.model = ADaMoSHiFiGANV1.from_pretrained(checkpoint_dir, local_files_only=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    def to(self, device: str) -> "ADaMoSBackend":
        if self.model is not None:
            self.model = self.model.to(device)
            self.device = device
        return self

    def enhance(self, waveform: torch.Tensor, input_sr: int) -> torch.Tensor:
        """waveform: [C, T] at input_sr → [C, T'] at input_sr."""
        assert self.model is not None, "Model not loaded"

        # Resample to model SR if needed
        if input_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, input_sr, self.sample_rate)

        # Process each channel independently (model is mono)
        channels = []
        for ch in range(waveform.shape[0]):
            mono = waveform[ch : ch + 1].unsqueeze(0).to(self.device).float()  # [1, 1, T]
            with torch.no_grad():
                mel = self.model.encode(mono)
                vocoded = self.model.decode(mel)
            # vocoded: [1, 1, T'] → [1, T']
            channels.append(vocoded.squeeze(0))
        result = torch.cat(channels, dim=0).cpu()  # [C, T']

        # Resample back if needed
        if input_sr != self.sample_rate:
            result = torchaudio.functional.resample(result, self.sample_rate, input_sr)

        return result

    def unload(self) -> None:
        self.model = None
        self.device = "cpu"


# ---------------------------------------------------------------------------
# BigVGAN v2 backend
# ---------------------------------------------------------------------------

class BigVGANBackend(VocoderBackend):
    """NVIDIA BigVGAN v2 — Snake activation, anti-aliased, full-band."""

    backend_type = "bigvgan"
    sample_rate = 44100

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self._checkpoint_dir = None

    def load(self, checkpoint_dir: str, device: str) -> None:
        try:
            import bigvgan as bigvgan_module
        except ImportError:
            raise ImportError(
                "BigVGAN package not installed. Run: pip install bigvgan\n"
                "See: https://github.com/NVIDIA/BigVGAN"
            )

        logger.info(f"[BigVGAN] Loading from {checkpoint_dir}")
        self._checkpoint_dir = checkpoint_dir

        # BigVGAN loads from HF-style directory or model name
        # Check for local generator file first
        gen_path = os.path.join(checkpoint_dir, "bigvgan_generator.pt")
        if os.path.isfile(gen_path):
            self.model = bigvgan_module.BigVGAN.from_pretrained(
                checkpoint_dir, use_cuda_kernel=False
            )
        else:
            # Try loading as HF model name
            self.model = bigvgan_module.BigVGAN.from_pretrained(
                checkpoint_dir, use_cuda_kernel=False
            )

        self.model.remove_weight_norm()
        self.model = self.model.eval().to(device)
        self.device = device
        self.sample_rate = self.model.h.sampling_rate
        logger.info(f"[BigVGAN] Loaded: sr={self.sample_rate}, device={device}")

    def to(self, device: str) -> "BigVGANBackend":
        if self.model is not None:
            self.model = self.model.to(device)
            self.device = device
        return self

    def enhance(self, waveform: torch.Tensor, input_sr: int) -> torch.Tensor:
        """waveform: [C, T] at input_sr → [C, T'] at input_sr."""
        assert self.model is not None, "Model not loaded"

        try:
            from bigvgan.meldataset import get_mel_spectrogram
        except ImportError:
            raise ImportError("BigVGAN meldataset module not found")

        # Resample to model SR if needed
        if input_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, input_sr, self.sample_rate)

        # Process each channel independently (BigVGAN is mono)
        channels = []
        for ch in range(waveform.shape[0]):
            mono = waveform[ch : ch + 1].to(self.device).float()  # [1, T]
            with torch.inference_mode():
                mel = get_mel_spectrogram(mono, self.model.h)  # [1, n_mels, T_frames]
                vocoded = self.model(mel)  # [1, 1, T']
            channels.append(vocoded.squeeze(0))  # [1, T']
        result = torch.cat(channels, dim=0).cpu()  # [C, T']

        # Resample back if needed
        if input_sr != self.sample_rate:
            result = torchaudio.functional.resample(result, self.sample_rate, input_sr)

        return result

    def unload(self) -> None:
        self.model = None
        self._checkpoint_dir = None
        self.device = "cpu"


# ---------------------------------------------------------------------------
# Vocos backend
# ---------------------------------------------------------------------------

class VocosBackend(VocoderBackend):
    """Vocos — iSTFT-based spectral-domain vocoder."""

    backend_type = "vocos"
    sample_rate = 44100

    def __init__(self):
        self.model = None
        self.device = "cpu"

    def load(self, checkpoint_dir: str, device: str) -> None:
        try:
            from vocos import Vocos
        except ImportError:
            raise ImportError(
                "Vocos package not installed. Run: pip install vocos\n"
                "See: https://github.com/gemelo-ai/vocos"
            )

        logger.info(f"[Vocos] Loading from {checkpoint_dir}")
        self.model = Vocos.from_hparams(os.path.join(checkpoint_dir, "config.yaml"))

        # Load weights
        ckpt_path = None
        for name in ("pytorch_model.bin", "model.pt", "vocos.pt"):
            p = os.path.join(checkpoint_dir, name)
            if os.path.isfile(p):
                ckpt_path = p
                break
        if ckpt_path is None:
            # Try safetensors
            for f in os.listdir(checkpoint_dir):
                if f.endswith(".safetensors"):
                    ckpt_path = os.path.join(checkpoint_dir, f)
                    break

        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.eval().to(device)
        self.device = device
        logger.info(f"[Vocos] Loaded: device={device}")

    def to(self, device: str) -> "VocosBackend":
        if self.model is not None:
            self.model = self.model.to(device)
            self.device = device
        return self

    def enhance(self, waveform: torch.Tensor, input_sr: int) -> torch.Tensor:
        """waveform: [C, T] at input_sr → [C, T'] at input_sr."""
        assert self.model is not None, "Model not loaded"

        # Resample to model SR if needed
        if input_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, input_sr, self.sample_rate)

        # Vocos can operate on features extracted from audio
        channels = []
        for ch in range(waveform.shape[0]):
            mono = waveform[ch : ch + 1].to(self.device).float()  # [1, T]
            with torch.inference_mode():
                # Vocos encode → decode roundtrip
                features = self.model.feature_extractor(mono)
                vocoded = self.model.decode(features)  # [1, T']
            channels.append(vocoded)
        result = torch.cat(channels, dim=0).cpu()  # [C, T']

        # Resample back if needed
        if input_sr != self.sample_rate:
            result = torchaudio.functional.resample(result, self.sample_rate, input_sr)

        return result

    def unload(self) -> None:
        self.model = None
        self.device = "cpu"


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY = {
    "adamos": ADaMoSBackend,
    "bigvgan": BigVGANBackend,
    "vocos": VocosBackend,
}


def _create_backend(vtype: str) -> VocoderBackend:
    cls = _BACKEND_REGISTRY.get(vtype)
    if cls is None:
        raise ValueError(f"Unknown vocoder type: {vtype!r}")
    return cls()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class VocoderService:
    """Manages vocoder model lifecycle and applies audio enhancement."""

    def __init__(self):
        self._backends: Dict[str, VocoderBackend] = {}
        self._vocoder_types: Dict[str, str] = {}  # model_name → vtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_available_vocoders(self) -> List[str]:
        """Scan checkpoints for directories that are actually vocoders.

        Returns a list like ``["None", "ADaMoSHiFiGANV1", "bigvgan_v2_44khz_128band_512x"]``.
        Only directories whose ``config.json`` matches a known vocoder
        signature are included — DiT, LM, and other model checkpoints
        are filtered out.
        """
        vocoders = ["None"]
        self._vocoder_types.clear()

        if not os.path.exists(VOCODER_DIR):
            return vocoders

        for item in sorted(os.listdir(VOCODER_DIR)):
            path = os.path.join(VOCODER_DIR, item)
            if not os.path.isdir(path):
                continue
            vtype = _detect_vocoder_type(path)
            if vtype is not None:
                vocoders.append(item)
                self._vocoder_types[item] = vtype
                logger.debug(f"[VocoderService] Found vocoder: {item} (type={vtype})")

        return vocoders

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_vocoder(self, model_name: str) -> VocoderBackend:
        """Load (or return cached) vocoder backend for *model_name*."""
        if model_name in self._backends:
            return self._backends[model_name]

        # Ensure we know the type
        if model_name not in self._vocoder_types:
            path = os.path.join(VOCODER_DIR, model_name)
            vtype = _detect_vocoder_type(path)
            if vtype is None:
                raise ValueError(
                    f"Checkpoint '{model_name}' is not a recognised vocoder. "
                    f"Known types: {list(_BACKEND_REGISTRY.keys())}"
                )
            self._vocoder_types[model_name] = vtype

        vtype = self._vocoder_types[model_name]
        backend = _create_backend(vtype)
        backend.load(os.path.join(VOCODER_DIR, model_name), self.device)
        self._backends[model_name] = backend
        logger.info(f"[VocoderService] Loaded {model_name} (type={vtype}) on {self.device}")
        return backend

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    @staticmethod
    def _frequency_blend(
        original: torch.Tensor,
        vocoded: torch.Tensor,
        sample_rate: int,
        crossover_hz: float = 6000.0,
        blend_width_hz: float = 2000.0,
    ) -> torch.Tensor:
        """Blend *original* and *vocoded* in the frequency domain.

        Below ``crossover_hz - blend_width_hz/2`` the original signal is
        kept untouched.  Above ``crossover_hz + blend_width_hz/2`` the
        vocoder output is used.  In between, a smooth cosine crossfade
        transitions from one to the other.

        Both tensors must be ``[C, T]`` at *sample_rate*.
        """
        # Ensure same length
        min_len = min(original.shape[-1], vocoded.shape[-1])
        original = original[..., :min_len]
        vocoded = vocoded[..., :min_len]

        n_fft = 2048
        hop = 512
        window = torch.hann_window(n_fft, device=original.device)

        channels = []
        for ch in range(original.shape[0]):
            # STFT both signals
            orig_stft = torch.stft(
                original[ch], n_fft, hop_length=hop,
                window=window, return_complex=True,
            )
            voco_stft = torch.stft(
                vocoded[ch], n_fft, hop_length=hop,
                window=window, return_complex=True,
            )

            # Build frequency-dependent blend mask [n_freq, 1]
            n_freq = orig_stft.shape[0]
            freqs = torch.linspace(0, sample_rate / 2, n_freq, device=original.device)

            low = crossover_hz - blend_width_hz / 2
            high = crossover_hz + blend_width_hz / 2

            # alpha=0 → use original, alpha=1 → use vocoder
            alpha = torch.zeros_like(freqs)
            in_transition = (freqs >= low) & (freqs <= high)
            alpha[in_transition] = 0.5 * (
                1.0 - torch.cos(
                    torch.pi * (freqs[in_transition] - low) / blend_width_hz
                )
            )
            alpha[freqs > high] = 1.0
            alpha = alpha.unsqueeze(-1)  # [n_freq, 1] for broadcasting

            # Blend in STFT domain
            blended_stft = (1.0 - alpha) * orig_stft + alpha * voco_stft

            # iSTFT back
            blended = torch.istft(
                blended_stft, n_fft, hop_length=hop,
                window=window, length=min_len,
            )
            channels.append(blended.unsqueeze(0))

        return torch.cat(channels, dim=0)

    def apply_vocoder(
        self,
        waveform: torch.Tensor,
        model_name: str,
        sample_rate: int = 48000,
        crossover_hz: float = 0.0,
        blend_width_hz: float = 2000.0,
    ) -> torch.Tensor:
        """Enhance *waveform* using the named vocoder.

        Args:
            waveform: Audio tensor. Accepted shapes:
                - ``[C, T]`` (channels, samples)
                - ``[B, C, T]`` (batch treated as channels)
                - ``[T]`` (mono)
            model_name: Name of vocoder checkpoint directory.
            sample_rate: Sample rate of *waveform*.
            crossover_hz: If > 0, keep original audio below this frequency
                and blend in vocoder output only above it.  Set to 0 to
                use the vocoder output fully (legacy behaviour).
            blend_width_hz: Width of the crossover transition band in Hz.
                A smooth cosine fade spans ±blend_width_hz/2 around
                *crossover_hz*.

        Returns:
            Enhanced waveform with same number of dimensions as input.
        """
        if not model_name or model_name.lower() in ("", "none", "null"):
            return waveform

        available = self.get_available_vocoders()
        if model_name not in available:
            logger.warning(
                f"[VocoderService] '{model_name}' not found in available vocoders "
                f"({available}). Returning original audio."
            )
            return waveform

        logger.info(f"[VocoderService] Applying vocoder '{model_name}' to audio")
        backend = self.load_vocoder(model_name)

        # Normalise shape to [C, T]
        original_dim = waveform.dim()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        elif waveform.dim() == 3:
            B, C, T = waveform.shape
            waveform = waveform.reshape(B * C, T)  # flatten batch+channels

        vocoded = backend.enhance(waveform, sample_rate)

        # Apply frequency-domain blending if crossover is set
        if crossover_hz > 0:
            logger.info(
                f"[VocoderService] Frequency blending: keeping original "
                f"below {crossover_hz}Hz, vocoder above "
                f"(transition: ±{blend_width_hz/2:.0f}Hz)"
            )
            result = self._frequency_blend(
                waveform, vocoded, sample_rate,
                crossover_hz=crossover_hz,
                blend_width_hz=blend_width_hz,
            )
        else:
            result = vocoded

        # Restore original dimensionality
        if original_dim == 1:
            result = result.squeeze(0)
        elif original_dim == 3:
            result = result.reshape(B, C, -1)

        return result


vocoder_service = VocoderService()

