from __future__ import annotations
import json
import statistics as st

def summarize_config(band, norm, fold_metrics):
    accs = [f["val_acc"] for f in fold_metrics]
    losses = [f["val_loss"] for f in fold_metrics]
    return {
        "band": band, "norm": norm, "folds": fold_metrics,
        "mean_acc": sum(accs)/len(accs), "std_acc": st.pstdev(accs) if len(accs) > 1 else 0.0,
        "mean_loss": sum(losses)/len(losses), "std_loss": st.pstdev(losses) if len(losses) > 1 else 0.0,
    }

def build_results(config_summaries, best_confusion, meta):
    return {"meta": meta, "configs": config_summaries, "best_confusion": best_confusion}

def write_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
