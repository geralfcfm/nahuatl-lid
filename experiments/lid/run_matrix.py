from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from . import config, train, evaluate, results
from .data import grouped_folds, SpecDataset
from .model import CRNN_LID


def _train_config(items: list, norm: str, folds: list, device: str, epochs: int, label: str = "",
                  model_cls: type[nn.Module] = CRNN_LID) -> tuple:
    """Train all folds for one (items, norm) config. Returns (fold_metrics, oof_labels, oof_preds)."""
    fold_metrics, oof_labels, oof_preds = [], [], []
    for i, (tr, va) in enumerate(folds, start=1):
        print(f"[{label}] fold {i}/{len(folds)}", flush=True)
        tr_items = [(items[j][0], items[j][1]) for j in tr]
        va_items = [(items[j][0], items[j][1]) for j in va]
        hist = train.train_fold(tr_items, va_items, norm, device=device, epochs=epochs,
                                label=f"{label} f{i}", model_cls=model_cls)
        best = int(np.argmin(hist["val_loss"]))
        fold_metrics.append({"val_acc": hist["val_acc"][best], "val_loss": hist["val_loss"][best]})
        m = model_cls().to(device)
        m.load_state_dict(hist["best_state"])
        m.eval()
        dl = DataLoader(SpecDataset(va_items, norm), batch_size=config.BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for x, y in dl:
                oof_preds += (m(x.to(device)) > 0).float().cpu().numpy().flatten().tolist()
                oof_labels += y.numpy().flatten().tolist()
    return fold_metrics, oof_labels, oof_preds


def _finalize(summaries, oof_by_config, epochs):
    best = max(summaries, key=lambda s: s["mean_acc"])
    bl, bp = oof_by_config[(best["band"], best["norm"])]
    best_confusion = evaluate.confusion_and_metrics(bl, bp)
    meta = {"seed": config.SEED, "k_folds": config.K_FOLDS, "epochs": epochs,
            "best_config": {"band": best["band"], "norm": best["norm"]}}
    return results.build_results(summaries, best_confusion, meta)


def run_matrix(items_by_band, filenames, device="cpu", epochs=config.EPOCHS, model_cls: type[nn.Module] = CRNN_LID):
    folds = grouped_folds(filenames, config.K_FOLDS)
    summaries, oof = [], {}
    for band, norm in config.CONFIGS:
        fm, ol, op = _train_config(items_by_band[band], norm, folds, device, epochs,
                                   label=f"{band}/{norm}", model_cls=model_cls)
        s = results.summarize_config(band, norm, fm)
        summaries.append(s)
        print(f"[{band}/{norm}] DONE mean_acc={s['mean_acc']:.4f} ± {s['std_acc']:.4f}", flush=True)
        oof[(band, norm)] = (ol, op)
    return _finalize(summaries, oof, epochs)


def run_bandwise(raw_items, filenames, bands, device="cpu", epochs=config.EPOCHS, model_cls: type[nn.Module] = CRNN_LID):
    """Memory-frugal: derive one band's features at a time from raw chunks, train its
    norm configs, then free before the next band. Returns the same shape as run_matrix."""
    from experiments.preprocess import features_for_band
    folds = grouped_folds(filenames, config.K_FOLDS)
    summaries, oof = [], {}
    for band in bands:
        print(f"=== band {band} ===", flush=True)
        band_items = features_for_band(raw_items, band)
        # norms for this band, derived from CONFIGS so the two runners can't diverge
        for norm in [n for (b, n) in config.CONFIGS if b == band]:
            fm, ol, op = _train_config(band_items, norm, folds, device, epochs,
                                       label=f"{band}/{norm}", model_cls=model_cls)
            s = results.summarize_config(band, norm, fm)
            summaries.append(s)
            print(f"[{band}/{norm}] DONE mean_acc={s['mean_acc']:.4f} ± {s['std_acc']:.4f}", flush=True)
            oof[(band, norm)] = (ol, op)
        del band_items
    return _finalize(summaries, oof, epochs)
