from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T
from . import config

def to_mono_16k(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr != config.SAMPLE_RATE:
        waveform = T.Resample(sr, config.SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform

def chunk_3s(waveform: torch.Tensor) -> list[torch.Tensor]:
    n = config.SAMPLES_PER_CHUNK
    total = waveform.shape[1]
    if total < n:
        return [F.pad(waveform, (0, n - total))]
    chunks = list(torch.split(waveform, n, dim=1))
    if chunks[-1].shape[1] != n:
        chunks = chunks[:-1]
    return chunks

def make_logmel(chunk: torch.Tensor, f_max: int, lowpass_hz: int | None) -> torch.Tensor:
    if lowpass_hz is not None:
        chunk = AF.lowpass_biquad(chunk, config.SAMPLE_RATE, cutoff_freq=lowpass_hz)
    mel = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP,
        f_min=config.F_MIN, f_max=f_max,
    )(chunk)
    return torch.log(mel + 1e-9)

def waveform_to_features(waveform: torch.Tensor, sr: int, band: str) -> list[torch.Tensor]:
    b = config.BANDS[band]
    wav = to_mono_16k(waveform, sr)
    return [make_logmel(c, b["f_max"], b["lowpass_hz"]) for c in chunk_3s(wav)]
