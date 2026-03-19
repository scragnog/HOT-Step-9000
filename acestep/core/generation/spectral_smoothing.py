"""
Spectral smoothing for anti-autotune post-processing.

Applies a 1D convolution [0.25, 0.50, 0.25] along the frequency axis
of the latent tensor to smooth out pitch quantization artifacts that
cause robotic/autotuned vocal quality.

The time axis is left untouched so drums and transients remain sharp.
"""

import torch
from typing import Dict, Any


def apply_spectral_smoothing(
    tensor: torch.Tensor,
    strength: float = 0.0,
) -> torch.Tensor:
    """Apply spectral smoothing along the frequency axis.

    Args:
        tensor: Latent tensor, shape [B, seq, dim] where dim ~ frequency.
        strength: 0.0 = no smoothing, 1.0 = full smoothing.

    Returns:
        Smoothed tensor (same shape).
    """
    if strength <= 0 or tensor is None:
        return tensor
    if tensor.dim() < 2:
        return tensor

    F = tensor.shape[-1]
    if F < 3:
        return tensor

    # kernel: [0.25, 0.50, 0.25] along frequency axis
    kernel = torch.tensor(
        [0.25, 0.5, 0.25],
        dtype=tensor.dtype,
        device=tensor.device,
    )

    # Flatten batch dims, apply conv1d along last axis
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, F)  # [N, F]
    flat = flat.unsqueeze(1)      # [N, 1, F]

    padded = torch.nn.functional.pad(flat, (1, 1), mode='reflect')
    kernel_w = kernel.view(1, 1, 3)
    smoothed = torch.nn.functional.conv1d(padded, kernel_w)
    smoothed = smoothed.squeeze(1).reshape(orig_shape)

    return (1.0 - strength) * tensor + strength * smoothed


def apply_spectral_smoothing_to_latent(
    latent: Dict[str, Any],
    strength: float = 0.0,
) -> Dict[str, Any]:
    """Convenience wrapper that accepts a latent dict with 'samples' key."""
    if strength <= 0 or latent is None:
        return latent
    if isinstance(latent, dict) and 'samples' in latent:
        latent['samples'] = apply_spectral_smoothing(latent['samples'], strength)
    elif isinstance(latent, torch.Tensor):
        return apply_spectral_smoothing(latent, strength)
    return latent
