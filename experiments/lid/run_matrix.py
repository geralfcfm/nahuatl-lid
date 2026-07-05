from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from . import config, train, evaluate, results
from .data import grouped_folds, group_id, SpecDataset
from .model import CRNN_LID, CNN_LID


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
    # Also emit the canonical wideband_16k/none confusion. The thesis quotes the
    # shuffle/normal accuracies at wideband_16k/none for every contrast, which is NOT
    # always the best config (e.g. the Raramuri time-shuffle best config is
    # lowpass_4k/instance), so best_confusion alone can misrepresent the quoted number.
    if ("wideband_16k", "none") in oof_by_config:
        wl, wp = oof_by_config[("wideband_16k", "none")]
        meta["wbnone_confusion"] = evaluate.confusion_and_metrics(wl, wp)
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


def run_bandwise_paired(raw_items, filenames, configs, device="cpu", epochs=config.EPOCHS):
    """Paired CRNN/CNN comparison: computes GroupKFold folds ONCE for `filenames`
    and trains BOTH CRNN_LID and CNN_LID on the SAME per-fold train/val splits, for
    each (band, norm) in `configs`. Because both architectures see identical folds
    within this single call, `crnn_folds[i]`/`cnn_folds[i]` are index-aligned per
    fold -> a paired t-test/TOST over them is valid (unlike separate `cv`/`cv_ablate`
    invocations, which each recompute the CV pair + folds independently).

    Returns {"meta": {...}, "configs": [{"band", "norm", "crnn_folds": [...],
    "cnn_folds": [...], "fold_group_ids": [[...], ...]}, ...]} -- NOT the same shape
    as run_matrix/run_bandwise (no single "model" per config, so no results.summarize_config).
    """
    from experiments.preprocess import features_for_band
    folds = grouped_folds(filenames, config.K_FOLDS)
    groups = np.array([group_id(f) for f in filenames])
    # Per-fold held-out group ids, independent of band/norm/arch -> lets anyone
    # re-derive and confirm both archs trained against the exact same splits.
    fold_group_ids = [sorted(set(groups[va].tolist())) for _, va in folds]

    band_cache: dict[str, list] = {}
    out_configs = []
    for band, norm in configs:
        if band not in band_cache:
            band_cache[band] = features_for_band(raw_items, band)
        items = band_cache[band]
        crnn_fm, _, _ = _train_config(items, norm, folds, device, epochs,
                                      label=f"{band}/{norm}/crnn", model_cls=CRNN_LID)
        cnn_fm, _, _ = _train_config(items, norm, folds, device, epochs,
                                     label=f"{band}/{norm}/cnn", model_cls=CNN_LID)
        crnn_accs = [f["val_acc"] for f in crnn_fm]
        cnn_accs = [f["val_acc"] for f in cnn_fm]
        print(f"[{band}/{norm}] paired DONE crnn_mean={sum(crnn_accs)/len(crnn_accs):.4f} "
              f"cnn_mean={sum(cnn_accs)/len(cnn_accs):.4f}", flush=True)
        out_configs.append({
            "band": band, "norm": norm,
            "crnn_folds": crnn_accs, "cnn_folds": cnn_accs,
            "fold_group_ids": fold_group_ids,
        })
    meta = {"seed": config.SEED, "k_folds": config.K_FOLDS, "epochs": epochs, "paired": True,
            "configs": [{"band": b, "norm": n} for b, n in configs]}
    return {"meta": meta, "configs": out_configs}


def run_paired_seeds(raw_items, filenames, band, norm, seeds, device="cpu", epochs=config.EPOCHS):
    """Seed-variance for the paired CNN/CRNN comparison on ONE (band, norm) config.

    Builds the GroupKFold folds and the band features ONCE (data held fixed) and trains
    BOTH architectures under each seed on the SAME folds, so the spread across `seeds`
    isolates training/initialisation non-determinism from data-draw and fold variance.
    Also captures per-epoch learning curves (train/val loss, val acc) for each arch on
    fold 1 under the first seed, for a convergence figure.

    Returns {"meta", "per_seed": [{"seed","crnn_folds","cnn_folds"}...],
    "fold_group_ids", "curves": {"crnn":{...}, "cnn":{...}}}.
    """
    from experiments.preprocess import features_for_band
    folds = grouped_folds(filenames, config.K_FOLDS)
    groups = np.array([group_id(f) for f in filenames])
    fold_group_ids = [sorted(set(groups[va].tolist())) for _, va in folds]
    items = features_for_band(raw_items, band)  # fixed across seeds
    orig_seed = config.SEED
    per_seed = []
    curves = None
    try:
        for si, s in enumerate(seeds):
            config.SEED = s  # train_fold re-seeds to config.SEED at the start of every fold
            crnn_fm, _, _ = _train_config(items, norm, folds, device, epochs,
                                          label=f"{band}/{norm}/crnn/seed{s}", model_cls=CRNN_LID)
            cnn_fm, _, _ = _train_config(items, norm, folds, device, epochs,
                                         label=f"{band}/{norm}/cnn/seed{s}", model_cls=CNN_LID)
            crnn_accs = [f["val_acc"] for f in crnn_fm]
            cnn_accs = [f["val_acc"] for f in cnn_fm]
            per_seed.append({"seed": s, "crnn_folds": crnn_accs, "cnn_folds": cnn_accs})
            print(f"[seed {s}] crnn_mean={sum(crnn_accs)/len(crnn_accs):.4f} "
                  f"cnn_mean={sum(cnn_accs)/len(cnn_accs):.4f}", flush=True)
            if si == 0:  # learning curves on fold 1 (both archs), first seed only
                tr, va = folds[0]
                tr_items = [(items[j][0], items[j][1]) for j in tr]
                va_items = [(items[j][0], items[j][1]) for j in va]
                h_crnn = train.train_fold(tr_items, va_items, norm, device=device, epochs=epochs,
                                          label="curve/crnn", model_cls=CRNN_LID)
                h_cnn = train.train_fold(tr_items, va_items, norm, device=device, epochs=epochs,
                                         label="curve/cnn", model_cls=CNN_LID)
                curves = {
                    "seed": s, "fold": 1, "epochs": epochs,
                    "crnn": {k: h_crnn[k] for k in ("train_loss", "val_loss", "val_acc")},
                    "cnn": {k: h_cnn[k] for k in ("train_loss", "val_loss", "val_acc")},
                }
    finally:
        config.SEED = orig_seed  # never leak the override, even on error
    meta = {"band": band, "norm": norm, "seeds": list(seeds), "k_folds": config.K_FOLDS,
            "epochs": epochs, "orig_seed": orig_seed}
    return {"meta": meta, "per_seed": per_seed, "fold_group_ids": fold_group_ids, "curves": curves}


def run_nested_cv(raw_items, filenames, band, norm, device="cpu", epochs=config.EPOCHS,
                  model_cls: type[nn.Module] = CRNN_LID, k_inner=4):
    """Nested, speaker-disjoint held-out TEST estimate for ONE (band, norm) config.

    Outer GroupKFold(K_FOLDS) holds out disjoint speakers as a TEST set. Within each
    outer-train split, an inner GroupKFold provides a validation split used ONLY for
    checkpoint (best-epoch) selection; the checkpointed model is then scored on the
    untouched outer-TEST speakers. The test speakers never influence the checkpoint,
    band, norm, threshold, or any other choice, so this is a genuine generalisation
    estimate -- unlike the single-split cross-validation *validation* accuracy reported
    elsewhere, where the checkpoint is chosen on the same fold that is then scored.

    Returns per-outer-fold test accuracy + the mean over outer folds. Single config,
    single seed, reported as a scoped sanity check (see the thesis limitation note).
    """
    from experiments.preprocess import features_for_band
    items = features_for_band(raw_items, band)
    groups = np.array([group_id(f) for f in filenames])
    outer = grouped_folds(filenames, config.K_FOLDS)
    test_accs, fold_group_ids = [], []
    for oi, (otr, ote) in enumerate(outer, start=1):
        config.seed_everything(config.SEED)
        otr_names = [filenames[j] for j in otr]
        inner = grouped_folds(otr_names, k_inner)
        itr_local, iva_local = inner[0]  # first inner fold: indices INTO otr_names
        itr = [int(otr[k]) for k in itr_local]
        iva = [int(otr[k]) for k in iva_local]
        tr_items = [(items[j][0], items[j][1]) for j in itr]
        va_items = [(items[j][0], items[j][1]) for j in iva]
        te_items = [(items[j][0], items[j][1]) for j in ote]
        # sanity: outer-test speakers disjoint from everything used for training/checkpoint
        assert set(groups[ote].tolist()).isdisjoint(set(groups[itr].tolist()) | set(groups[iva].tolist())), \
            "nested-CV leakage: test speakers overlap train/val"
        h = train.train_fold(tr_items, va_items, norm, device=device, epochs=epochs,
                             label=f"nested o{oi}/{band}/{norm}", model_cls=model_cls)
        m = model_cls().to(device)
        m.load_state_dict(h["best_state"])
        m.eval()
        dl = DataLoader(SpecDataset(te_items, norm), batch_size=config.BATCH_SIZE, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                out = m(x)
                correct += ((out > 0).float() == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        test_accs.append(acc)
        fold_group_ids.append(sorted(set(groups[ote].tolist())))
        print(f"[nested o{oi}] test_acc={acc:.4f} (test groups={len(fold_group_ids[-1])})", flush=True)
    mean = sum(test_accs) / len(test_accs)
    print(f"[nested {band}/{norm}] DONE test_mean={mean:.4f}", flush=True)
    meta = {"band": band, "norm": norm, "seed": config.SEED, "k_outer": config.K_FOLDS,
            "k_inner": k_inner, "epochs": epochs,
            "protocol": "nested-CV: outer speaker-disjoint TEST, inner GroupKFold val for checkpoint only"}
    return {"meta": meta, "test_accs": test_accs, "fold_group_ids": fold_group_ids}
