from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .model import CRNN_LID
from .data import SpecDataset
from . import config


def train_fold(
    train_items: list, val_items: list, normalize: str, device: str = "cpu", epochs: int = config.EPOCHS,
    label: str = "", model_cls: type[nn.Module] = CRNN_LID,
) -> dict:
    config.seed_everything(config.SEED)
    train_dl = DataLoader(SpecDataset(train_items, normalize), batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(SpecDataset(val_items, normalize), batch_size=config.BATCH_SIZE, shuffle=False)

    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    crit = nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())

    for ep in range(epochs):
        model.train()
        tl = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tl += loss.item()
        model.eval()
        vl, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                out = model(x)
                vl += crit(out, y).item()
                correct += ((out > 0).float() == y).sum().item()
                total += y.size(0)
        avg_tl = tl / max(len(train_dl), 1)
        avg_vl = vl / max(len(val_dl), 1)
        acc = correct / max(total, 1)
        sched.step(avg_vl)
        print(f"  [{label}] ep {ep + 1}/{epochs} "
              f"train_loss={avg_tl:.4f} val_loss={avg_vl:.4f} val_acc={acc:.4f}", flush=True)
        hist["train_loss"].append(avg_tl)
        hist["val_loss"].append(avg_vl)
        hist["val_acc"].append(acc)
        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            best_state = copy.deepcopy(model.state_dict())

    hist["best_state"] = best_state
    hist["best_val_loss"] = best_val_loss
    return hist
