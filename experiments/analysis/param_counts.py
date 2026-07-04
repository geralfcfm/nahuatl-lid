"""Task A: trainable parameter counts for CRNN_LID vs CNN_LID.

Instantiates both models from experiments/lid/model.py (architecture only --
no weights loaded, no training) and counts trainable parameters. Writes
results/param_counts.json and results/param_counts.md (self-contained,
canonical output), and ALSO appends the same section to
results/paired_stats.md for convenience when reading that file standalone.

Run order matters: run `paired_stats.py` FIRST, then `param_counts.py`
SECOND. `paired_stats.py` opens results/paired_stats.md in "w" mode
(truncate + rewrite), so it would silently drop an already-appended param
section if run after this script. `param_counts.py` re-appends its section
idempotently every time it runs, so re-running paired_stats.py alone and
then re-running this script always restores both files to a consistent
state -- but results/param_counts.md is the safe, standalone source of
truth if you only need the param counts and don't want to re-run
paired_stats.py at all.

Run with:
    uv run python3 -m experiments.analysis.paired_stats   # first
    uv run python3 -m experiments.analysis.param_counts   # second
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.lid.model import CNN_LID, CRNN_LID

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    crnn = CRNN_LID()
    cnn = CNN_LID()

    crnn_params = count_trainable_params(crnn)
    cnn_params = count_trainable_params(cnn)
    ratio = crnn_params / cnn_params

    def fp32_mib(n_params: int) -> float:
        return (n_params * 4) / (1024 * 1024)

    out = {
        "CRNN_LID": {
            "trainable_params": crnn_params,
            "fp32_size_mib": round(fp32_mib(crnn_params), 4),
        },
        "CNN_LID": {
            "trainable_params": cnn_params,
            "fp32_size_mib": round(fp32_mib(cnn_params), 4),
        },
        "ratio_crnn_over_cnn": round(ratio, 4),
        "note": (
            "Trainable-parameter counts (requires_grad=True) from freshly instantiated, "
            "untrained model.py architectures -- no checkpoints loaded, no training performed."
        ),
    }

    out_json = RESULTS_DIR / "param_counts.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_json}")
    print(json.dumps(out, indent=2))

    section_lines = []
    section_lines.append("## Model parameter counts\n")
    section_lines.append(
        "Instantiated from `experiments/lid/model.py` (untrained, architecture-only "
        "count of `requires_grad=True` parameters).\n"
    )
    section_lines.append("| Model | Trainable params | Approx. fp32 size (MiB) |")
    section_lines.append("|---|---:|---:|")
    section_lines.append(f"| CRNN_LID | {crnn_params:,} | {fp32_mib(crnn_params):.3f} |")
    section_lines.append(f"| CNN_LID | {cnn_params:,} | {fp32_mib(cnn_params):.3f} |")
    section_lines.append(f"\nRatio CRNN/CNN = **{ratio:.2f}x**.\n")

    # Self-contained, canonical output: safe to regenerate any time, in any
    # order relative to paired_stats.py, without depending on the state of
    # results/paired_stats.md.
    standalone_path = RESULTS_DIR / "param_counts.md"
    with open(standalone_path, "w") as f:
        f.write("# Model parameter counts (Task A)\n\n")
        f.write("\n".join(section_lines) + "\n")
    print(f"wrote {standalone_path}")

    # Convenience copy appended to paired_stats.md. NOTE: paired_stats.py
    # opens paired_stats.md in "w" mode, so this section is dropped if
    # paired_stats.py is re-run afterward -- always re-run this script
    # (param_counts.py) AFTER paired_stats.py to restore it. See module
    # docstring for the canonical run order.
    md_path = RESULTS_DIR / "paired_stats.md"
    with open(md_path, "a") as f:
        f.write("\n" + "\n".join(section_lines) + "\n")
    print(f"appended param counts to {md_path}")


if __name__ == "__main__":
    main()
