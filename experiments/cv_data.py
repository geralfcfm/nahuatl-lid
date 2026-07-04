from __future__ import annotations

import csv
import os
import random
import torch
import soundfile as sf
from .lid import audio


MIRROR_LANGS: set[str] = {"es", "en", "gn"}  # present in fsicoli/common_voice_22_0 with a train split

_LABEL_NAMES: dict[str, str] = {"es": "spanish", "en": "english"}


def label_name_for(contrast_lang: str) -> str:
    """Filename prefix for the contrast class (must NOT start with 'nahuatl')."""
    return _LABEL_NAMES.get(contrast_lang, contrast_lang)


def speaker_index_map(client_ids: list[str]) -> dict[str, int]:
    """Deterministic: sorted unique client_ids → 0..n-1."""
    return {cid: i for i, cid in enumerate(sorted(set(client_ids)))}


def cv_filename(lang: str, speaker_idx: int, clip_idx: int) -> str:
    return f"{lang}_{speaker_idx}_{clip_idx}.pt"


def read_validated_rows(tsv_path: str) -> list[tuple[str, str, str]]:
    """Return (client_id, path, accents) per validated row."""
    rows: list[tuple[str, str, str]] = []
    with open(tsv_path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((r["client_id"], r["path"], r.get("accents", "") or ""))
    return rows


def _wave_to_chunks(wf: torch.Tensor, sr: int) -> list[torch.Tensor]:
    return audio.chunk_3s(audio.to_mono_16k(wf, sr))


def local_cv_raw(cv_dir: str, label_value: float, label_name: str) -> list[tuple[torch.Tensor, float, str]]:
    """Read a local Common Voice download (validated.tsv + clips/) → raw 16k mono chunks,
    labeled label_value, filenames '{label_name}_{speaker_idx}_{clip_idx}.pt'."""
    rows = read_validated_rows(os.path.join(cv_dir, "validated.tsv"))
    spk = speaker_index_map([c for c, _, _ in rows])
    items: list[tuple[torch.Tensor, float, str]] = []
    per_spk: dict[int, int] = {}
    for client_id, path, _ in rows:
        data, sr = sf.read(os.path.join(cv_dir, "clips", path), dtype="float32", always_2d=True)
        wf = torch.from_numpy(data.T).contiguous()
        si = spk[client_id]
        for c in _wave_to_chunks(wf, sr):
            ci = per_spk.get(si, 0)
            per_spk[si] = ci + 1
            items.append((c, label_value, cv_filename(label_name, si, ci)))
    return items


def nahuatl_cv_raw(ncx_dir: str) -> list[tuple[torch.Tensor, float, str]]:
    return local_cv_raw(ncx_dir, 1.0, "nahuatl")


def balance_pair(
    nah: list[tuple[torch.Tensor, float, str]],
    contrast: list[tuple[torch.Tensor, float, str]],
    seed: int = 42,
) -> tuple[list[tuple[torch.Tensor, float, str]], list[tuple[torch.Tensor, float, str]]]:
    """Sub-sample both lists to the smaller length, shuffled (seeded) to keep speaker spread."""
    n = min(len(nah), len(contrast))
    rng = random.Random(seed)
    nah = nah[:]
    contrast = contrast[:]
    rng.shuffle(nah)
    rng.shuffle(contrast)
    return nah[:n], contrast[:n]


def hf_cv_raw(
    lang: str,
    label_name: str,
    target_count: int,
    mexican_only: bool = False,
) -> list[tuple[torch.Tensor, float, str]]:
    from datasets import load_dataset  # lazy; needs datasets<3.0
    ds = load_dataset(
        "fsicoli/common_voice_22_0",
        lang,
        split="train",  # mirror has no "validated" split; train is the large validated subset
        trust_remote_code=True,
        streaming=True,
    )
    spk: dict[str, int] = {}
    per_spk_counter: dict[int, int] = {}
    items: list[tuple[torch.Tensor, float, str]] = []
    for it in ds:
        if len(items) >= target_count:
            break
        if mexican_only and "éxico" not in (it.get("accents") or ""):
            continue
        cid = it["client_id"]
        if cid not in spk:
            spk[cid] = len(spk)
        si = spk[cid]
        wf = torch.tensor(it["audio"]["array"], dtype=torch.float32).unsqueeze(0)
        for c in _wave_to_chunks(wf, it["audio"]["sampling_rate"]):
            if len(items) >= target_count:
                break
            ci = per_spk_counter.get(si, 0)
            per_spk_counter[si] = ci + 1
            items.append((c, 0.0, cv_filename(label_name, si, ci)))
    return items


def build_cv_pair(
    ncx_dir: str,
    contrast_lang: str,
    contrast_dir: str | None = None,
) -> tuple[list[tuple[torch.Tensor, float, str]], list[str]]:
    nah = nahuatl_cv_raw(ncx_dir)
    name = label_name_for(contrast_lang)
    if contrast_dir is not None:                      # user-downloaded CV 26.0 language
        contrast = local_cv_raw(contrast_dir, 0.0, name)
    else:                                             # streamable mirror language
        # NOTE: no Mexican-accent filter — the streaming HF mirror's `accents` field is
        # empty, so filtering yielded 0 Spanish (degenerate all-Nahuatl run). Plain `es`
        # is a valid Spanish contrast for the language-discrimination question.
        contrast = hf_cv_raw(contrast_lang, name, target_count=len(nah), mexican_only=False)
    nah, contrast = balance_pair(nah, contrast)
    raw = nah + contrast
    return raw, [fn for _, _, fn in raw]
