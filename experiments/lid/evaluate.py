from __future__ import annotations


def confusion_and_metrics(labels, preds) -> dict:
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + fn + fp + tn, 1)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
