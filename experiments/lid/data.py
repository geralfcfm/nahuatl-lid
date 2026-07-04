from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

def group_id(filename: str) -> str:
    # Group id = "<lang>_<source-file-idx>"; matches the names minted in preprocess.decode_corpus.
    parts = os.path.basename(filename).split("_")
    return f"{parts[0]}_{parts[1]}"

def label_of(filename: str) -> float:
    return 1.0 if os.path.basename(filename).startswith("nahuatl") else 0.0

class SpecDataset(Dataset):
    def __init__(self, items, normalize: str = "instance"):
        assert normalize in ("none", "instance")
        self.items = items
        self.normalize = normalize

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        spec, label = self.items[idx]
        if self.normalize == "instance":
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec, torch.tensor(label, dtype=torch.float32)

def grouped_folds(filenames, k: int):
    groups = np.array([group_id(f) for f in filenames])
    labels = np.array([label_of(f) for f in filenames])
    dummy = np.zeros(len(filenames))
    folds = []
    for tr, va in GroupKFold(n_splits=k).split(dummy, labels, groups=groups):
        assert set(groups[tr]).isdisjoint(set(groups[va])), "GroupKFold leakage"
        folds.append((tr, va))
    return folds
