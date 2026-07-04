from __future__ import annotations

import torch
import torchaudio.transforms as T


def shuffle_time(spec: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly permute the time axis (last dim) of a (C, F, T) log-mel spectrogram.

    Destroys temporal/phonotactic order while preserving the per-frame spectral content.
    """
    t = spec.shape[-1]
    perm = torch.randperm(t, generator=generator)
    return spec[..., perm]


def degrade_chunk(
    chunk: torch.Tensor,
    sr: int = 16000,
    down_sr: int = 8000,
    noise_snr_db: float = 20.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Simulate a low-quality mic: band-limit by resampling sr->down_sr->sr (caps content at
    down_sr/2), then add Gaussian noise at the given SNR.

    Returns a chunk at the original sr/length.
    """
    down = T.Resample(sr, down_sr)(chunk)
    back = T.Resample(down_sr, sr)(down)
    # match length to the input (resample round-trip can be off by a sample)
    n = chunk.shape[-1]
    if back.shape[-1] < n:
        back = torch.nn.functional.pad(back, (0, n - back.shape[-1]))
    back = back[..., :n]
    sig_power = back.pow(2).mean().clamp_min(1e-12)
    noise_power = sig_power / (10 ** (noise_snr_db / 10))
    noise = torch.randn(back.shape, generator=generator) * noise_power.sqrt()
    return back + noise
