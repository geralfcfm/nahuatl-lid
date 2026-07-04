"""Task A: paired fold-level re-analysis of already-committed `results/*.json`.

NO new model training happens here. This script only reads the committed
JSON result files and derives paired-fold statistics (mean diff, sample SD,
t-test, exact Wilcoxon, t-based 95% CI, and a TOST equivalence verdict at a
+/-1 pp margin) plus per-config t-based 95% CIs of the mean accuracy.

Outputs: results/paired_stats.json (machine-readable) and
results/paired_stats.md (human-readable tables). Run with:

    uv run python3 -m experiments.analysis.paired_stats
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.analysis.stats_util import (
    EQUIV_MARGIN_PP,
    HAVE_SCIPY,
    SCIPY_VERSION,
    T_CRIT_90_DF4,
    T_CRIT_95_DF4,
    mean_ci_t,
    paired_compare,
    welch_compare,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# (filename, human label) for every experiment file whose full 3-band x 2-norm
# matrix appears in Appendix.tex's "Full per-configuration accuracy tables"
# (app:full-tables). results_cv_gn.json (Guarani) is committed but is
# explicitly excluded from all thesis tables/figures as "inconclusive"
# (Appendix.tex line 18-21); it is reported separately below, not folded
# into the norm/bandwidth comparisons that mirror the thesis appendix.
EXPERIMENT_FILES: list[tuple[str, str]] = [
    ("results.json", "Baseline (field Nahuatl vs. studio Spanish)"),
    ("results_shuffle.json", "Probe: time-shuffle on baseline"),
    ("results_degrade.json", "Probe: cheap-mic degradation on baseline"),
    ("results_cv_es.json", "CV: Nahuatl vs. Spanish (es)"),
    ("results_cv_en.json", "CV: Nahuatl vs. English (en)"),
    ("results_cv_tar.json", "CV: Nahuatl vs. Raramuri (tar)"),
    ("results_cv_sei.json", "CV: Nahuatl vs. Seri (sei)"),
    ("results_cv_qxp.json", "CV: Nahuatl vs. Quechua (qxp)"),
    ("results_cv_en_cnn.json", "Architecture ablation: CNN front-end, en contrast"),
    ("results_cv_en_shuffle.json", "Probe: genuine time-shuffle, en contrast"),
]
EXCLUDED_FILE = ("results_cv_gn.json", "CV: Nahuatl vs. Guarani (gn) -- thesis-excluded, inconclusive")

BANDS = ["wideband_16k", "bandrestrict_4k", "lowpass_4k"]
NORMS = ["none", "instance"]

# Bandwidth comparison (item b): only the five contrast languages, per brief.
BANDWIDTH_FILES = [
    ("results_cv_es.json", "es"),
    ("results_cv_en.json", "en"),
    ("results_cv_tar.json", "tar"),
    ("results_cv_sei.json", "sei"),
    ("results_cv_qxp.json", "qxp"),
]


def load(fname: str) -> dict:
    with open(RESULTS_DIR / fname) as f:
        return json.load(f)


def get_config(data: dict, band: str, norm: str) -> dict:
    for c in data["configs"]:
        if c["band"] == band and c["norm"] == norm:
            return c
    raise KeyError(f"config band={band} norm={norm} not found")


def fold_accs(data: dict, band: str, norm: str) -> list[float]:
    cfg = get_config(data, band, norm)
    return [f["val_acc"] for f in cfg["folds"]]


def build_per_config_ci() -> list[dict]:
    rows = []
    for fname, label in EXPERIMENT_FILES + [EXCLUDED_FILE]:
        data = load(fname)
        for band in BANDS:
            for norm in NORMS:
                accs = fold_accs(data, band, norm)
                row = mean_ci_t(accs, label=f"{label} | {band}/{norm}")
                row["file"] = fname
                row["band"] = band
                row["norm"] = norm
                row["reported_pop_sd_pp"] = round(get_config(data, band, norm)["std_acc"] * 100.0, 4)
                rows.append(row)
    return rows


def build_baseline_config_invariance() -> list[dict]:
    """(a) Every config in results.json vs. the best config, paired by fold."""
    data = load("results.json")
    best = data["meta"]["best_config"]
    best_accs = fold_accs(data, best["band"], best["norm"])
    out = []
    for c in data["configs"]:
        label = f"{c['band']}/{c['norm']} vs best ({best['band']}/{best['norm']})"
        accs = fold_accs(data, c["band"], c["norm"])
        res = paired_compare(accs, best_accs, label).to_dict()
        res["file"] = "results.json"
        res["config"] = f"{c['band']}/{c['norm']}"
        res["best_config"] = f"{best['band']}/{best['norm']}"
        res["is_best_itself"] = c["band"] == best["band"] and c["norm"] == best["norm"]
        out.append(res)
    return out


def build_bandwidth() -> list[dict]:
    """(b) wideband_16k/none vs lowpass_4k/none, per contrast. Diff reported as
    wideband - lowpass (pp lost by going to the true low-pass condition)."""
    out = []
    for fname, tag in BANDWIDTH_FILES:
        data = load(fname)
        wide = fold_accs(data, "wideband_16k", "none")
        low = fold_accs(data, "lowpass_4k", "none")
        label = f"{tag}: wideband_16k/none - lowpass_4k/none"
        res = paired_compare(wide, low, label).to_dict()
        res["file"] = fname
        res["contrast"] = tag
        out.append(res)
    return out


def build_norm() -> list[dict]:
    """(c) none vs instance, for all 3 bands x every experiment file (matches
    the full set of tables in Appendix.tex's app:full-tables, minus the
    thesis-excluded Guarani contrast)."""
    out = []
    for fname, label in EXPERIMENT_FILES:
        data = load(fname)
        for band in BANDS:
            none_accs = fold_accs(data, band, "none")
            inst_accs = fold_accs(data, band, "instance")
            cmp_label = f"{label} | {band}: none - instance"
            res = paired_compare(none_accs, inst_accs, cmp_label).to_dict()
            res["file"] = fname
            res["experiment"] = label
            res["band"] = band
            out.append(res)
    return out


def build_architecture() -> list[dict]:
    """(d) CNN vs CRNN on the English contrast, wideband_16k/none.

    Fold alignment CANNOT be confirmed from committed JSON: results_cv_en.json
    (CRNN) and results_cv_en_cnn.json (CNN) come from two SEPARATE calls to
    experiments/cv_data.py:build_cv_pair() (see experiments/modal_app.py,
    `experiment="cv"` vs `experiment="cv_ablate"`). For contrast="en",
    build_cv_pair streams the contrast side from the HF mirror
    fsicoli/common_voice_22_0 (experiments/cv_data.py:hf_cv_raw) with no
    pinned dataset revision and no persisted per-fold file manifest in the
    committed results/*.json. GroupKFold itself is a deterministic function
    of (file order, k), but nothing in the repo proves the two runs saw the
    same file order. Per the brief, this is therefore reported as an
    UNPAIRED (Welch) comparison, not a paired one.
    """
    crnn = load("results_cv_en.json")
    cnn = load("results_cv_en_cnn.json")
    crnn_accs = fold_accs(crnn, "wideband_16k", "none")
    cnn_accs = fold_accs(cnn, "wideband_16k", "none")
    label = "CRNN (results_cv_en.json) - CNN (results_cv_en_cnn.json), wideband_16k/none"
    res = welch_compare(crnn_accs, cnn_accs, label).to_dict()
    res["file_a"] = "results_cv_en.json (CRNN)"
    res["file_b"] = "results_cv_en_cnn.json (CNN)"
    res["alignment_confirmed"] = False
    res["alignment_note"] = (
        "Fold-level alignment between the two runs could NOT be confirmed from committed "
        "data (separate build_cv_pair() calls; contrast side streamed from an HF mirror "
        "with no pinned revision or persisted file manifest). Reported as unpaired Welch "
        "t-test/CI, per brief instructions for the unconfirmed-alignment case."
    )
    return [res]


PAIRED_FILE = "results_cv_en_paired.json"


def build_architecture_paired() -> list[dict]:
    """(d-paired) CRNN vs. CNN on the English contrast, wideband_16k/none, from a
    SINGLE paired run (experiments/lid/run_matrix.run_bandwise_paired, invoked via
    experiments/modal_app.py experiment="cv_paired").

    Unlike build_architecture()'s unpaired Welch comparison, here GroupKFold folds
    are computed ONCE and BOTH architectures train on the identical per-fold splits
    within one process, so fold index i is the same held-out group set for both ->
    a valid PAIRED t-test/TOST. The logged `fold_group_ids` make the shared
    partition auditable. This supersedes (d) for the architecture claim.

    Returns [] if the paired result file is absent (it is produced only after the
    cv_paired experiment is run), so this script still runs from the originally
    committed result files alone.
    """
    path = RESULTS_DIR / PAIRED_FILE
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    cfg = next(
        (c for c in data["configs"] if c["band"] == "wideband_16k" and c["norm"] == "none"),
        None,
    )
    if cfg is None:
        return []
    crnn_accs = cfg["crnn_folds"]
    cnn_accs = cfg["cnn_folds"]
    label = "CRNN - CNN, English contrast, wideband_16k/none (single paired run)"
    res = paired_compare(crnn_accs, cnn_accs, label).to_dict()
    res["file"] = PAIRED_FILE
    res["alignment_confirmed"] = True
    res["alignment_note"] = (
        "Fold-level pairing IS confirmed: run_bandwise_paired computes the GroupKFold "
        "folds once and trains both architectures on those identical splits within a "
        "single run; the per-fold held-out group ids are logged in `fold_group_ids`. "
        "This supersedes the unpaired Welch estimate in (d) for the architecture "
        "comparison. NOTE: this is a fresh training run, so its CRNN/CNN accuracies "
        "are NOT identical to the separately-run results_cv_en.json / "
        "results_cv_en_cnn.json headline numbers."
    )
    res["crnn_mean_pct"] = round(sum(crnn_accs) / len(crnn_accs) * 100.0, 4)
    res["cnn_mean_pct"] = round(sum(cnn_accs) / len(cnn_accs) * 100.0, 4)
    res["seed"] = data["meta"].get("seed")
    res["k_folds"] = data["meta"].get("k_folds")
    res["epochs"] = data["meta"].get("epochs")
    return [res]


def main() -> None:
    per_config_ci = build_per_config_ci()
    baseline = build_baseline_config_invariance()
    bandwidth = build_bandwidth()
    norm = build_norm()
    architecture = build_architecture()
    architecture_paired = build_architecture_paired()

    out = {
        "meta": {
            "n_folds": 5,
            "df_paired": 4,
            "t_crit_95_df4": T_CRIT_95_DF4,
            "t_crit_90_df4": T_CRIT_90_DF4,
            "equivalence_margin_pp": EQUIV_MARGIN_PP,
            "stats_backend": "scipy" if HAVE_SCIPY else "exact_t",
            "scipy_version": SCIPY_VERSION,
            "sd_convention_note": (
                "results/*.json 'std_acc' is POPULATION SD (statistics.pstdev, ddof=0). "
                "All SD-based quantities in THIS file (sample_sd_diff_pp, sample_sd_pp, CIs) "
                "use SAMPLE SD (ddof=1) instead, as required for t-based inference."
            ),
            "tost_rule_note": (
                "TOST at alpha=.05 against a +/-1pp margin is implemented as: 90% CI of the "
                "difference (using t_0.95(4)=2.132) lies entirely within [-1, +1] pp. The 95% "
                "CI (t_0.975(4)=2.776) is reported separately for uncertainty only and is NOT "
                "the TOST criterion."
            ),
            "independence_caveat": (
                "All paired comparisons are fold-level, EXPLORATORY comparisons over 5 "
                "overlapping-training GroupKFold splits of the SAME data (same seed=42, "
                "k_folds=5) -- NOT five independent experiments, NOT population-level proof. "
                "Treat p-values/CIs here as descriptive/exploratory, not confirmatory."
            ),
            "no_new_training": "All numbers derive from already-committed results/*.json; no models were (re)trained.",
        },
        "per_config_ci": per_config_ci,
        "comparisons": {
            "baseline_config_invariance": baseline,
            "bandwidth_wideband_minus_lowpass": bandwidth,
            "norm_none_minus_instance": norm,
            "architecture_cnn_vs_crnn": architecture,
            "architecture_cnn_vs_crnn_paired": architecture_paired,
        },
    }

    out_json = RESULTS_DIR / "paired_stats.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_json}")

    write_markdown(out)


def fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def fmt_p(p: float | None) -> str:
    return f"{p:.4f}" if p is not None else "n/a"


def write_markdown(out: dict) -> None:
    meta = out["meta"]
    lines: list[str] = []
    lines.append("# Paired fold statistics -- re-analysis of committed results\n")
    lines.append(
        "Source: `experiments/analysis/paired_stats.py`, reading already-committed "
        "`results/*.json` only. **No new model training.**\n"
    )
    lines.append(f"- Stats backend: **{meta['stats_backend']}**" + (f" (scipy {meta['scipy_version']})" if meta["scipy_version"] else "") + "\n")
    lines.append(f"- n = 5 folds per config, df = 4 for all paired comparisons; t₀.₉₇₅(4) = {meta['t_crit_95_df4']}, t₀.₉₅(4) = {meta['t_crit_90_df4']}\n")
    lines.append(f"- {meta['sd_convention_note']}\n")
    lines.append(f"- **TOST rule:** {meta['tost_rule_note']}\n")
    lines.append(f"- **Independence caveat:** {meta['independence_caveat']}\n")

    lines.append("\n## (a) Baseline config-invariance (each config vs. best, `results/results.json`)\n")
    lines.append("| Config | vs. best | diff (pp) | 95% CI (pp) | t p | Wilcoxon p | 90% CI (pp) | TOST @±1pp |")
    lines.append("|---|---|---:|---|---:|---:|---|---|")
    for r in out["comparisons"]["baseline_config_invariance"]:
        tag = " (best itself)" if r["is_best_itself"] else ""
        lines.append(
            f"| {r['config']}{tag} | {r['best_config']} | {r['diff_mean_pp']:+.4f} | {fmt_ci(r['ci95_pp'])} | "
            f"{r['p_ttest_two_sided']:.4f} | {fmt_p(r['p_wilcoxon_two_sided'])} | {fmt_ci(r['ci90_pp'])} | "
            f"{'ESTABLISHED' if r['tost_established'] else 'not established'} |"
        )

    lines.append("\n## (b) Bandwidth: wideband_16k/none − lowpass_4k/none (pp lost at 4kHz low-pass), paired by fold\n")
    lines.append("| Contrast | diff (pp) | 95% CI (pp) | t p | Wilcoxon p | 90% CI (pp) | TOST @±1pp |")
    lines.append("|---|---:|---|---:|---:|---|---|")
    for r in out["comparisons"]["bandwidth_wideband_minus_lowpass"]:
        lines.append(
            f"| {r['contrast']} | {r['diff_mean_pp']:+.4f} | {fmt_ci(r['ci95_pp'])} | {r['p_ttest_two_sided']:.4f} | "
            f"{fmt_p(r['p_wilcoxon_two_sided'])} | {fmt_ci(r['ci90_pp'])} | {'ESTABLISHED' if r['tost_established'] else 'not established'} |"
        )
    lines.append(
        "\nSign convention: wideband − lowpass (pp lost by the 4 kHz low-pass); positive = "
        "low-pass lower. Note: qxp is slightly positive, i.e. it does not follow a naive "
        "channel-only expectation.\n"
    )

    lines.append("\n## (c) Instance norm: none − instance, all 3 bands × every experiment (paired by fold)\n")
    lines.append("| Experiment | Band | diff (pp) | 95% CI (pp) | t p | Wilcoxon p | 90% CI (pp) | TOST @±1pp |")
    lines.append("|---|---|---:|---|---:|---:|---|---|")
    for r in out["comparisons"]["norm_none_minus_instance"]:
        lines.append(
            f"| {r['experiment']} | {r['band']} | {r['diff_mean_pp']:+.4f} | {fmt_ci(r['ci95_pp'])} | "
            f"{r['p_ttest_two_sided']:.4f} | {fmt_p(r['p_wilcoxon_two_sided'])} | {fmt_ci(r['ci90_pp'])} | "
            f"{'ESTABLISHED' if r['tost_established'] else 'not established'} |"
        )
    n_norm = len(out["comparisons"]["norm_none_minus_instance"])
    n_est = sum(1 for r in out["comparisons"]["norm_none_minus_instance"] if r["tost_established"])
    lines.append(f"\n{n_est}/{n_norm} none-vs-instance cells establish equivalence at ±1pp (TOST, 90% CI).\n")
    lines.append(
        "`results_cv_gn.json` (Guarani) is committed data but is excluded here, matching the thesis's own "
        "treatment of that contrast as inconclusive (Appendix.tex: \"wildly unstable fold-to-fold behaviour ... "
        "excluded from all tables and figures\"). Its per-config CIs are still listed in the per-config table below, "
        "labeled thesis-excluded, but no comparison is drawn from it.\n"
    )

    lines.append("\n## (d) Architecture: CRNN vs. CNN, English contrast, wideband_16k/none\n")
    for r in out["comparisons"]["architecture_cnn_vs_crnn"]:
        lines.append(f"- **Paired?** {r['paired']} -- {r['alignment_note']}")
        lines.append(f"- diff (CRNN − CNN) = **{r['diff_mean_pp']:+.4f} pp**")
        lines.append(f"- Welch df ≈ {r['df']:.2f}, 95% CI (pp) = {fmt_ci(r['ci95_pp'])}, t p = {r['p_ttest_two_sided']:.4f}")
        lines.append(f"- 90% CI (pp) = {fmt_ci(r['ci90_pp'])} -> TOST @±1pp: {'ESTABLISHED' if r['tost_established'] else 'not established'}")
        lines.append(f"- sample SD: CRNN={r['sample_sd_a_pp']:.4f} pp (n={r['n1']}), CNN={r['sample_sd_b_pp']:.4f} pp (n={r['n2']})")

    paired_arch = out["comparisons"].get("architecture_cnn_vs_crnn_paired", [])
    if paired_arch:
        lines.append("\n## (d-paired) Architecture: CRNN vs. CNN, English contrast, wideband_16k/none — SINGLE PAIRED RUN\n")
        for r in paired_arch:
            lines.append(f"- **Paired?** {r['paired']} (alignment confirmed: {r['alignment_confirmed']}) — {r['alignment_note']}")
            lines.append(f"- run: seed={r['seed']}, k_folds={r['k_folds']}, epochs={r['epochs']}; CRNN mean = **{r['crnn_mean_pct']:.4f}%**, CNN mean = **{r['cnn_mean_pct']:.4f}%**")
            lines.append(f"- diff (CRNN − CNN) = **{r['diff_mean_pp']:+.4f} pp** (sample SD of diff {r['sample_sd_diff_pp']:.4f} pp, SE {r['se_diff_pp']:.4f} pp)")
            lines.append(f"- 95% CI (pp) = {fmt_ci(r['ci95_pp'])}, paired-t p = {r['p_ttest_two_sided']:.4f}, Wilcoxon p = {fmt_p(r['p_wilcoxon_two_sided'])} ({r['wilcoxon_method']})")
            lines.append(f"- 90% CI (pp) = {fmt_ci(r['ci90_pp'])} → TOST @±1pp: {'ESTABLISHED' if r['tost_established'] else 'not established'}")

    lines.append("\n## Per-config t-based 95% CI of the mean accuracy (all configs, all files)\n")
    lines.append("Replaces population-SD-only language; CI computed as mean ± t₀.₉₇₅(4)·(sample SD)/√5.\n")
    lines.append("| File | Band/Norm | Mean acc (%) | Sample SD (pp) | 95% CI of mean (%) | Reported pop. SD (pp) |")
    lines.append("|---|---|---:|---:|---|---:|")
    for r in out["per_config_ci"]:
        lines.append(
            f"| {r['file']} | {r['band']}/{r['norm']} | {r['mean_acc_pct']:.4f} | {r['sample_sd_pp']:.4f} | "
            f"[{r['ci95_low_pct']:.4f}, {r['ci95_high_pct']:.4f}] | {r['reported_pop_sd_pp']:.4f} |"
        )

    with open(RESULTS_DIR / "paired_stats.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {RESULTS_DIR / 'paired_stats.md'}")


if __name__ == "__main__":
    main()
