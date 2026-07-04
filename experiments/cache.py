from __future__ import annotations

import json
import os
from typing import Callable

import torch


def is_cached(cache_dir: str) -> bool:
    return os.path.exists(os.path.join(cache_dir, "manifest.json"))


def save_corpus(
    raw_items: list[tuple[torch.Tensor, float, str]],
    filenames: list[str],
    bands: list[str],
    cache_dir: str,
    features_for_band: Callable[
        [list[tuple[torch.Tensor, float, str]], str],
        list[tuple[torch.Tensor, float, str]],
    ],
) -> None:
    """Persist raw chunks + per-band Mel + a manifest to cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    labels = [lbl for _, lbl, _ in raw_items]
    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump({"filenames": filenames, "labels": labels, "bands": bands}, f)
    torch.save([c for c, _, _ in raw_items], os.path.join(cache_dir, "raw.pt"))
    for b in bands:
        feats = features_for_band(raw_items, b)
        torch.save([spec for spec, _, _ in feats], os.path.join(cache_dir, f"mel_{b}.pt"))


def load_manifest(cache_dir: str) -> tuple[list[str], list[float]]:
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        m = json.load(f)
    return m["filenames"], m["labels"]


def load_band_items(cache_dir: str, band: str) -> list[tuple[torch.Tensor, float, str]]:
    filenames, labels = load_manifest(cache_dir)
    specs = torch.load(os.path.join(cache_dir, f"mel_{band}.pt"), weights_only=True)
    return list(zip(specs, labels, filenames))


def load_raw_items(cache_dir: str) -> list[tuple[torch.Tensor, float, str]]:
    filenames, labels = load_manifest(cache_dir)
    chunks = torch.load(os.path.join(cache_dir, "raw.pt"), weights_only=True)
    return list(zip(chunks, labels, filenames))
