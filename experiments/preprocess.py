from __future__ import annotations

import glob, os
import soundfile as sf
import torch
from .lid import audio, config


def _load_waveform(path: str) -> tuple[torch.Tensor, int]:
    """Decode an audio file to a (channels, frames) float tensor + sample rate
    via soundfile (libsndfile decodes MP3; avoids torchaudio.load/torchcodec)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (frames, channels)
    return torch.from_numpy(data.T).contiguous(), sr


def _raw_chunks(wf: torch.Tensor, sr: int) -> list[torch.Tensor]:
    """Mono, 16 kHz, 3 s raw waveform chunks (no Mel yet)."""
    return audio.chunk_3s(audio.to_mono_16k(wf, sr))


def decode_corpus(
    audio_dir: str,
    max_files: int | None = None,
    max_chunks_per_file: int | None = None,
) -> tuple[list[tuple[torch.Tensor, float, str]], list[str]]:
    """Decode Nahuatl + (balanced) Spanish ONCE into raw 16 kHz mono 3 s chunks.
    Returns (raw_items, filenames) where raw_items = (raw_chunk, label, filename)."""
    raw: list[tuple[torch.Tensor, float, str]] = []
    paths = sorted(glob.glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True))
    if max_files is not None:
        paths = paths[:max_files]
    for idx, p in enumerate(paths):
        if idx % 10 == 0:
            print(f"decode: nahuatl {idx + 1}/{len(paths)} files", flush=True)
        wf, sr = _load_waveform(p)
        chunks = _raw_chunks(wf, sr)
        if max_chunks_per_file is not None:
            chunks = chunks[:max_chunks_per_file]
        for j, c in enumerate(chunks):
            raw.append((c, 1.0, f"nahuatl_{idx}_{j}.pt"))
    print(f"decode: Nahuatl done -> {len(raw)} chunks; target Spanish={len(raw)}", flush=True)

    target = len(raw)  # balance Spanish to the Nahuatl chunk count
    from datasets import load_dataset  # lazy: only needed at preprocessing time (Modal)
    ds = load_dataset("ciempiess/ciempiess_light", split="train")
    count = 0
    for idx, it in enumerate(ds):
        if count >= target:
            break
        wf = torch.tensor(it["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        chunks = _raw_chunks(wf, it["audio"]["sampling_rate"])
        if max_chunks_per_file is not None:
            chunks = chunks[:max_chunks_per_file]
        for j, c in enumerate(chunks):
            if count >= target:
                break
            raw.append((c, 0.0, f"spanish_{idx}_{j}.pt"))
            count += 1
            if count % 2000 == 0:
                print(f"decode: spanish {count}/{target} chunks", flush=True)
    filenames = [fn for _, _, fn in raw]
    print(f"decode: corpus ready -> {len(raw)} total chunks ({len(filenames)} filenames)", flush=True)
    return raw, filenames


def features_for_band(
    raw_items: list[tuple[torch.Tensor, float, str]], band: str
) -> list[tuple[torch.Tensor, float, str]]:
    """Apply the band's Mel transform to each raw chunk → (logmel, label, filename)."""
    b = config.BANDS[band]
    return [(audio.make_logmel(c, b["f_max"], b["lowpass_hz"]), label, fn) for c, label, fn in raw_items]
