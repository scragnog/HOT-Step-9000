"""A/B vocoder comparison tool.

Runs an existing ACE-Step wav through each available vocoder backend,
saves side-by-side outputs, and logs objective quality metrics.

Usage
-----
    python -m acestep.tools.vocoder_comparison --input path/to/song.wav
    python -m acestep.tools.vocoder_comparison --input path/to/song.wav --device cpu
    python -m acestep.tools.vocoder_comparison --input path/to/song.wav --download-bigvgan
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from loguru import logger


def _download_bigvgan(dest: str) -> None:
    """Download BigVGAN v2 44kHz from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    model_id = "nvidia/bigvgan_v2_44khz_128band_512x"
    logger.info(f"Downloading {model_id} → {dest}")
    snapshot_download(
        repo_id=model_id,
        local_dir=dest,
        ignore_patterns=["*.md", "*.txt", "bigvgan_discriminator_optimizer.pt"],
    )
    logger.info(f"Download complete: {dest}")


def _compute_metrics(
    reference: torch.Tensor, enhanced: torch.Tensor, sr: int
) -> dict:
    """Compute basic objective quality metrics between reference and enhanced audio.

    Both tensors should be [C, T] at the same sample rate.
    """
    metrics = {}

    # Ensure same length
    min_len = min(reference.shape[-1], enhanced.shape[-1])
    ref = reference[..., :min_len].float()
    enh = enhanced[..., :min_len].float()

    # SNR (signal-to-noise ratio of the difference)
    noise = enh - ref
    signal_power = (ref ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power > 0:
        metrics["snr_db"] = 10 * torch.log10(signal_power / noise_power).item()
    else:
        metrics["snr_db"] = float("inf")

    # RMS levels
    metrics["ref_rms"] = ref.pow(2).mean().sqrt().item()
    metrics["enh_rms"] = enh.pow(2).mean().sqrt().item()

    # Peak levels
    metrics["ref_peak"] = ref.abs().max().item()
    metrics["enh_peak"] = enh.abs().max().item()

    # Multi-resolution STFT distance (simplified)
    stft_distances = []
    for n_fft in [512, 1024, 2048, 4096]:
        hop = n_fft // 4
        # Average over channels
        for ch in range(ref.shape[0]):
            ref_stft = torch.stft(
                ref[ch], n_fft, hop_length=hop, return_complex=True, window=torch.hann_window(n_fft)
            )
            enh_stft = torch.stft(
                enh[ch], n_fft, hop_length=hop, return_complex=True, window=torch.hann_window(n_fft)
            )
            # Spectral convergence
            sc = torch.norm(ref_stft.abs() - enh_stft.abs()) / (torch.norm(ref_stft.abs()) + 1e-7)
            stft_distances.append(sc.item())
    metrics["avg_spectral_convergence"] = sum(stft_distances) / len(stft_distances)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compare vocoder backends on ACE-Step audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input wav file")
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: same dir as input)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--download-bigvgan", action="store_true",
        help="Download BigVGAN v2 44kHz checkpoint before running",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoints dir
    project_root = Path(__file__).resolve().parent.parent.parent
    ckpt_dir = project_root / "checkpoints"

    # Optionally download BigVGAN
    if args.download_bigvgan:
        bigvgan_dir = ckpt_dir / "bigvgan_v2_44khz_128band_512x"
        if not bigvgan_dir.exists():
            _download_bigvgan(str(bigvgan_dir))
        else:
            logger.info(f"BigVGAN already present at {bigvgan_dir}")

    # Load input audio
    import soundfile as sf
    import numpy as np
    wav_np, sr = sf.read(str(input_path), dtype="float32")
    if wav_np.ndim == 1:
        wav_np = wav_np[np.newaxis, :]  # [1, T]
    else:
        wav_np = wav_np.T  # [T, C] → [C, T]
    waveform = torch.from_numpy(wav_np)

    # Discover vocoders
    from acestep.core.vocoder_service import VocoderService

    service = VocoderService()
    service.device = args.device
    available = service.get_available_vocoders()

    vocoder_names = [v for v in available if v.lower() != "none"]
    if not vocoder_names:
        logger.error("No vocoder models found in checkpoints/")
        sys.exit(1)

    logger.info(f"Found vocoders: {vocoder_names}")
    stem = input_path.stem

    results = {}
    for name in vocoder_names:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing: {name}")
        logger.info(f"{'=' * 60}")

        try:
            # Measure VRAM before
            if torch.cuda.is_available() and args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                vram_before = torch.cuda.memory_allocated() / 1024 ** 3

            start = time.perf_counter()
            enhanced = service.apply_vocoder(waveform.clone(), name, sr)
            elapsed = time.perf_counter() - start

            # Measure VRAM after
            vram_peak = 0.0
            if torch.cuda.is_available() and args.device == "cuda":
                vram_peak = torch.cuda.max_memory_allocated() / 1024 ** 3

            # Save output
            out_path = output_dir / f"{stem}_{name}.wav"
            sf.write(str(out_path), enhanced.numpy().T, sr)
            logger.info(f"  Saved: {out_path}")

            # Compute metrics
            metrics = _compute_metrics(waveform, enhanced, sr)
            metrics["time_sec"] = elapsed
            metrics["vram_peak_gb"] = vram_peak
            results[name] = metrics

            logger.info(f"  Time: {elapsed:.2f}s")
            logger.info(f"  VRAM peak: {vram_peak:.2f} GB")
            logger.info(f"  SNR: {metrics['snr_db']:.1f} dB")
            logger.info(f"  Spectral convergence: {metrics['avg_spectral_convergence']:.4f}")
            logger.info(f"  Peak: {metrics['ref_peak']:.4f} → {metrics['enh_peak']:.4f}")

            # Unload to free VRAM for next model
            if name in service._backends:
                service._backends[name].unload()
                del service._backends[name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    # Summary table
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"{'Model':<45} {'Time':>6} {'SNR':>8} {'SpConv':>8} {'VRAM':>6}")
    logger.info("-" * 75)
    for name, m in results.items():
        if "error" in m:
            logger.info(f"{name:<45} {'ERROR':>6}")
        else:
            logger.info(
                f"{name:<45} {m['time_sec']:>5.1f}s {m['snr_db']:>7.1f}dB "
                f"{m['avg_spectral_convergence']:>7.4f} {m['vram_peak_gb']:>5.2f}G"
            )

    logger.info(f"\nOutput files in: {output_dir}")


if __name__ == "__main__":
    main()
