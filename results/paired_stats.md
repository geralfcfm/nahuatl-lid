# Paired fold statistics -- re-analysis of committed results

Source: `experiments/analysis/paired_stats.py`, reading already-committed `results/*.json` only. **No new model training.**

- Stats backend: **scipy** (scipy 1.13.1)

- n = 5 folds per config, df = 4 for all paired comparisons; t₀.₉₇₅(4) = 2.776, t₀.₉₅(4) = 2.132

- results/*.json 'std_acc' is POPULATION SD (statistics.pstdev, ddof=0). All SD-based quantities in THIS file (sample_sd_diff_pp, sample_sd_pp, CIs) use SAMPLE SD (ddof=1) instead, as required for t-based inference.

- **TOST rule:** TOST at alpha=.05 against a +/-1pp margin is implemented as: 90% CI of the difference (using t_0.95(4)=2.132) lies entirely within [-1, +1] pp. The 95% CI (t_0.975(4)=2.776) is reported separately for uncertainty only and is NOT the TOST criterion.

- **Independence caveat:** All paired comparisons are fold-level, EXPLORATORY comparisons over 5 overlapping-training GroupKFold splits of the SAME data (same seed=42, k_folds=5) -- NOT five independent experiments, NOT population-level proof. Treat p-values/CIs here as descriptive/exploratory, not confirmatory.


## (a) Baseline config-invariance (each config vs. best, `results/results.json`)

| Config | vs. best | diff (pp) | 95% CI (pp) | t p | Wilcoxon p | 90% CI (pp) | TOST @±1pp |
|---|---|---:|---|---:|---:|---|---|
| wideband_16k/none | lowpass_4k/none | -0.0067 | [-0.063, +0.050] | 0.7572 | 1.0000 | [-0.050, +0.037] | ESTABLISHED |
| wideband_16k/instance | lowpass_4k/none | -0.0101 | [-0.053, +0.033] | 0.5529 | 0.8750 | [-0.043, +0.023] | ESTABLISHED |
| bandrestrict_4k/none | lowpass_4k/none | -0.0168 | [-0.059, +0.025] | 0.3262 | 0.5000 | [-0.049, +0.015] | ESTABLISHED |
| bandrestrict_4k/instance | lowpass_4k/none | -0.0101 | [-0.051, +0.031] | 0.5291 | 0.5000 | [-0.041, +0.021] | ESTABLISHED |
| lowpass_4k/none (best itself) | lowpass_4k/none | +0.0000 | [+0.000, +0.000] | 1.0000 | n/a | [+0.000, +0.000] | ESTABLISHED |
| lowpass_4k/instance | lowpass_4k/none | -0.0269 | [-0.083, +0.029] | 0.2560 | 0.5000 | [-0.070, +0.016] | ESTABLISHED |

## (b) Bandwidth: wideband_16k/none − lowpass_4k/none (pp lost at 4kHz low-pass), paired by fold

| Contrast | diff (pp) | 95% CI (pp) | t p | Wilcoxon p | 90% CI (pp) | TOST @±1pp |
|---|---:|---|---:|---:|---|---|
| es | +0.2590 | [-1.422, +1.940] | 0.6909 | 0.8125 | [-1.032, +1.550] | not established |
| en | +0.3128 | [-1.562, +2.188] | 0.6673 | 0.4375 | [-1.127, +1.752] | not established |
| tar | +1.5128 | [-2.625, +5.651] | 0.3675 | 0.6250 | [-1.664, +4.690] | not established |
| sei | -0.1739 | [-5.257, +4.910] | 0.9289 | 0.8125 | [-4.077, +3.729] | not established |
| qxp | +0.1613 | [-5.301, +5.623] | 0.9386 | 1.0000 | [-4.033, +4.355] | not established |

Sign convention: wideband − lowpass (pp lost by the 4 kHz low-pass); positive = low-pass lower. Note: qxp is slightly positive, i.e. it does not follow a naive channel-only expectation.


## (c) Instance norm: none − instance, all 3 bands × every experiment (paired by fold)

| Experiment | Band | diff (pp) | 95% CI (pp) | t p | Wilcoxon p | 90% CI (pp) | TOST @±1pp |
|---|---|---:|---|---:|---:|---|---|
| Baseline (field Nahuatl vs. studio Spanish) | wideband_16k | +0.0034 | [-0.014, +0.021] | 0.6213 | 1.0000 | [-0.010, +0.017] | ESTABLISHED |
| Baseline (field Nahuatl vs. studio Spanish) | bandrestrict_4k | -0.0067 | [-0.055, +0.041] | 0.7174 | 0.8750 | [-0.044, +0.030] | ESTABLISHED |
| Baseline (field Nahuatl vs. studio Spanish) | lowpass_4k | +0.0269 | [-0.029, +0.083] | 0.2560 | 0.5000 | [-0.016, +0.070] | ESTABLISHED |
| Probe: time-shuffle on baseline | wideband_16k | +0.0034 | [-0.034, +0.041] | 0.8149 | 1.0000 | [-0.025, +0.032] | ESTABLISHED |
| Probe: time-shuffle on baseline | bandrestrict_4k | +0.0538 | [-0.006, +0.114] | 0.0669 | 0.0625 | [+0.008, +0.100] | ESTABLISHED |
| Probe: time-shuffle on baseline | lowpass_4k | +0.0067 | [-0.062, +0.075] | 0.7990 | 0.8750 | [-0.046, +0.059] | ESTABLISHED |
| Probe: cheap-mic degradation on baseline | wideband_16k | -0.1077 | [-0.208, -0.007] | 0.0414 | 0.0625 | [-0.185, -0.030] | ESTABLISHED |
| Probe: cheap-mic degradation on baseline | bandrestrict_4k | -0.0269 | [-0.103, +0.049] | 0.3821 | 0.6250 | [-0.085, +0.032] | ESTABLISHED |
| Probe: cheap-mic degradation on baseline | lowpass_4k | -0.0639 | [-0.157, +0.029] | 0.1281 | 0.1250 | [-0.135, +0.007] | ESTABLISHED |
| CV: Nahuatl vs. Spanish (es) | wideband_16k | +1.4071 | [+0.679, +2.135] | 0.0058 | 0.0625 | [+0.848, +1.966] | not established |
| CV: Nahuatl vs. Spanish (es) | bandrestrict_4k | +1.1531 | [-1.217, +3.523] | 0.2481 | 0.3125 | [-0.667, +2.973] | not established |
| CV: Nahuatl vs. Spanish (es) | lowpass_4k | +0.6205 | [-0.242, +1.483] | 0.1166 | 0.1250 | [-0.042, +1.283] | not established |
| CV: Nahuatl vs. English (en) | wideband_16k | +1.1336 | [-1.271, +3.538] | 0.2608 | 0.3125 | [-0.713, +2.980] | not established |
| CV: Nahuatl vs. English (en) | bandrestrict_4k | +3.2197 | [-0.603, +7.042] | 0.0795 | 0.1250 | [+0.285, +6.155] | not established |
| CV: Nahuatl vs. English (en) | lowpass_4k | +2.2085 | [-1.040, +5.457] | 0.1321 | 0.0625 | [-0.286, +4.703] | not established |
| CV: Nahuatl vs. Raramuri (tar) | wideband_16k | +1.2822 | [-1.300, +3.865] | 0.2401 | 0.1875 | [-0.701, +3.265] | not established |
| CV: Nahuatl vs. Raramuri (tar) | bandrestrict_4k | +1.1858 | [-0.635, +3.006] | 0.1448 | 0.1875 | [-0.212, +2.584] | not established |
| CV: Nahuatl vs. Raramuri (tar) | lowpass_4k | -1.0837 | [-4.277, +2.109] | 0.3994 | 0.6250 | [-3.535, +1.368] | not established |
| CV: Nahuatl vs. Seri (sei) | wideband_16k | +2.3266 | [-5.679, +10.332] | 0.4650 | 0.6250 | [-3.821, +8.474] | not established |
| CV: Nahuatl vs. Seri (sei) | bandrestrict_4k | +3.9678 | [+0.406, +7.529] | 0.0365 | 0.0625 | [+1.233, +6.702] | not established |
| CV: Nahuatl vs. Seri (sei) | lowpass_4k | +1.7987 | [-2.852, +6.449] | 0.3433 | 0.4375 | [-1.772, +5.370] | not established |
| CV: Nahuatl vs. Quechua (qxp) | wideband_16k | +0.0440 | [-5.269, +5.357] | 0.9828 | 1.0000 | [-4.036, +4.124] | not established |
| CV: Nahuatl vs. Quechua (qxp) | bandrestrict_4k | +0.6254 | [-4.597, +5.848] | 0.7562 | 1.0000 | [-3.384, +4.635] | not established |
| CV: Nahuatl vs. Quechua (qxp) | lowpass_4k | -0.3469 | [-2.115, +1.421] | 0.6150 | 0.6250 | [-1.705, +1.011] | not established |
| Architecture ablation: CNN front-end, en contrast | wideband_16k | +1.2995 | [-0.222, +2.821] | 0.0767 | 0.1250 | [+0.131, +2.468] | not established |
| Architecture ablation: CNN front-end, en contrast | bandrestrict_4k | +1.5536 | [+0.116, +2.991] | 0.0399 | 0.0625 | [+0.450, +2.657] | not established |
| Architecture ablation: CNN front-end, en contrast | lowpass_4k | +0.4396 | [-0.438, +1.317] | 0.2365 | 0.4375 | [-0.234, +1.113] | not established |
| Probe: genuine time-shuffle, en contrast | wideband_16k | +2.0862 | [+0.195, +3.978] | 0.0376 | 0.0625 | [+0.634, +3.539] | not established |
| Probe: genuine time-shuffle, en contrast | bandrestrict_4k | +1.1577 | [-5.435, +7.751] | 0.6514 | 0.8125 | [-3.905, +6.220] | not established |
| Probe: genuine time-shuffle, en contrast | lowpass_4k | +1.8224 | [-0.844, +4.489] | 0.1306 | 0.1250 | [-0.225, +3.870] | not established |

9/30 none-vs-instance cells establish equivalence at ±1pp (TOST, 90% CI).

`results_cv_gn.json` (Guarani) is committed data but is excluded here, matching the thesis's own treatment of that contrast as inconclusive (Appendix.tex: "wildly unstable fold-to-fold behaviour ... excluded from all tables and figures"). Its per-config CIs are still listed in the per-config table below, labeled thesis-excluded, but no comparison is drawn from it.


## (d) Architecture: CRNN vs. CNN, English contrast, wideband_16k/none

- **Paired?** False -- Fold-level alignment between the two runs could NOT be confirmed from committed data (separate build_cv_pair() calls; contrast side streamed from an HF mirror with no pinned revision or persisted file manifest). Reported as unpaired Welch t-test/CI, per brief instructions for the unconfirmed-alignment case.
- diff (CRNN − CNN) = **+0.0733 pp**
- Welch df ≈ 7.99, 95% CI (pp) = [-2.016, +2.163], t p = 0.9375
- 90% CI (pp) = [-1.612, +1.758] -> TOST @±1pp: not established
- sample SD: CRNN=1.4134 pp (n=5), CNN=1.4514 pp (n=5)

## Per-config t-based 95% CI of the mean accuracy (all configs, all files)

Replaces population-SD-only language; CI computed as mean ± t₀.₉₇₅(4)·(sample SD)/√5.

| File | Band/Norm | Mean acc (%) | Sample SD (pp) | 95% CI of mean (%) | Reported pop. SD (pp) |
|---|---|---:|---:|---|---:|
| results.json | wideband_16k/none | 99.9764 | 0.0439 | [99.9220, 100.0309] | 0.0392 |
| results.json | wideband_16k/instance | 99.9731 | 0.0349 | [99.9298, 100.0164] | 0.0312 |
| results.json | bandrestrict_4k/none | 99.9664 | 0.0357 | [99.9220, 100.0107] | 0.0319 |
| results.json | bandrestrict_4k/instance | 99.9731 | 0.0255 | [99.9414, 100.0048] | 0.0228 |
| results.json | lowpass_4k/none | 99.9832 | 0.0206 | [99.9576, 100.0088] | 0.0184 |
| results.json | lowpass_4k/instance | 99.9563 | 0.0439 | [99.9018, 100.0107] | 0.0392 |
| results_shuffle.json | wideband_16k/none | 99.9092 | 0.1181 | [99.7625, 100.0558] | 0.1057 |
| results_shuffle.json | wideband_16k/instance | 99.9058 | 0.1181 | [99.7591, 100.0525] | 0.1057 |
| results_shuffle.json | bandrestrict_4k/none | 99.9058 | 0.0941 | [99.7889, 100.0227] | 0.0842 |
| results_shuffle.json | bandrestrict_4k/instance | 99.8520 | 0.1060 | [99.7203, 99.9836] | 0.0948 |
| results_shuffle.json | lowpass_4k/none | 99.9293 | 0.0550 | [99.8610, 99.9977] | 0.0492 |
| results_shuffle.json | lowpass_4k/instance | 99.9226 | 0.0803 | [99.8229, 100.0224] | 0.0719 |
| results_degrade.json | wideband_16k/none | 99.7577 | 0.1603 | [99.5587, 99.9568] | 0.1434 |
| results_degrade.json | wideband_16k/instance | 99.8654 | 0.1084 | [99.7308, 100.0000] | 0.0969 |
| results_degrade.json | bandrestrict_4k/none | 99.8318 | 0.1265 | [99.6747, 99.9888] | 0.1131 |
| results_degrade.json | bandrestrict_4k/instance | 99.8587 | 0.0710 | [99.7705, 99.9468] | 0.0635 |
| results_degrade.json | lowpass_4k/none | 99.7645 | 0.2005 | [99.5155, 100.0134] | 0.1793 |
| results_degrade.json | lowpass_4k/instance | 99.8284 | 0.1548 | [99.6361, 100.0206] | 0.1385 |
| results_cv_es.json | wideband_16k/none | 96.8830 | 1.7761 | [94.6778, 99.0883] | 1.5886 |
| results_cv_es.json | wideband_16k/instance | 95.4759 | 1.5531 | [93.5476, 97.4043] | 1.3891 |
| results_cv_es.json | bandrestrict_4k/none | 95.8374 | 1.6832 | [93.7474, 97.9273] | 1.5055 |
| results_cv_es.json | bandrestrict_4k/instance | 94.6843 | 1.7584 | [92.5010, 96.8676] | 1.5727 |
| results_cv_es.json | lowpass_4k/none | 96.6240 | 1.4571 | [94.8148, 98.4332] | 1.3033 |
| results_cv_es.json | lowpass_4k/instance | 96.0035 | 1.1430 | [94.5843, 97.4228] | 1.0223 |
| results_cv_en.json | wideband_16k/none | 97.0296 | 1.4134 | [95.2746, 98.7845] | 1.2642 |
| results_cv_en.json | wideband_16k/instance | 95.8960 | 0.6970 | [95.0305, 96.7615] | 0.6234 |
| results_cv_en.json | bandrestrict_4k/none | 95.7543 | 1.3251 | [94.1090, 97.3997] | 1.1852 |
| results_cv_en.json | bandrestrict_4k/instance | 92.5347 | 4.2257 | [87.2878, 97.7815] | 3.7795 |
| results_cv_en.json | lowpass_4k/none | 96.7168 | 1.4358 | [94.9340, 98.4996] | 1.2842 |
| results_cv_en.json | lowpass_4k/instance | 94.5083 | 3.9256 | [89.6340, 99.3826] | 3.5112 |
| results_cv_tar.json | wideband_16k/none | 97.7573 | 1.8309 | [95.4840, 100.0306] | 1.6376 |
| results_cv_tar.json | wideband_16k/instance | 96.4751 | 3.8927 | [91.6417, 101.3085] | 3.4817 |
| results_cv_tar.json | bandrestrict_4k/none | 94.6777 | 3.9158 | [89.8155, 99.5398] | 3.5024 |
| results_cv_tar.json | bandrestrict_4k/instance | 93.4919 | 5.1695 | [87.0732, 99.9106] | 4.6237 |
| results_cv_tar.json | lowpass_4k/none | 96.2445 | 4.9988 | [90.0376, 102.4513] | 4.4711 |
| results_cv_tar.json | lowpass_4k/instance | 97.3282 | 2.5686 | [94.1389, 100.5174] | 2.2974 |
| results_cv_sei.json | wideband_16k/none | 89.2421 | 7.8080 | [79.5472, 98.9370] | 6.9837 |
| results_cv_sei.json | wideband_16k/instance | 86.9155 | 7.2163 | [77.9552, 95.8758] | 6.4545 |
| results_cv_sei.json | bandrestrict_4k/none | 90.0331 | 8.9249 | [78.9514, 101.1149] | 7.9827 |
| results_cv_sei.json | bandrestrict_4k/instance | 86.0654 | 10.2218 | [73.3734, 98.7574] | 9.1426 |
| results_cv_sei.json | lowpass_4k/none | 89.4160 | 6.7383 | [81.0493, 97.7826] | 6.0269 |
| results_cv_sei.json | lowpass_4k/instance | 87.6172 | 8.0950 | [77.5659, 97.6685] | 7.2404 |
| results_cv_qxp.json | wideband_16k/none | 82.1285 | 12.4844 | [66.6271, 97.6299] | 11.1664 |
| results_cv_qxp.json | wideband_16k/instance | 82.0845 | 10.1809 | [69.4432, 94.7258] | 9.1061 |
| results_cv_qxp.json | bandrestrict_4k/none | 86.2127 | 6.8760 | [77.6750, 94.7504] | 6.1501 |
| results_cv_qxp.json | bandrestrict_4k/instance | 85.5873 | 6.3603 | [77.6899, 93.4846] | 5.6888 |
| results_cv_qxp.json | lowpass_4k/none | 81.9672 | 13.5576 | [65.1333, 98.8012] | 12.1263 |
| results_cv_qxp.json | lowpass_4k/instance | 82.3141 | 12.4258 | [66.8854, 97.7427] | 11.1139 |
| results_cv_en_cnn.json | wideband_16k/none | 96.9562 | 1.4514 | [95.1540, 98.7584] | 1.2982 |
| results_cv_en_cnn.json | wideband_16k/instance | 95.6567 | 1.4541 | [93.8512, 97.4622] | 1.3006 |
| results_cv_en_cnn.json | bandrestrict_4k/none | 94.5867 | 2.9656 | [90.9044, 98.2690] | 2.6525 |
| results_cv_en_cnn.json | bandrestrict_4k/instance | 93.0331 | 3.1632 | [89.1054, 96.9608] | 2.8293 |
| results_cv_en_cnn.json | lowpass_4k/none | 95.7201 | 2.9799 | [92.0201, 99.4201] | 2.6653 |
| results_cv_en_cnn.json | lowpass_4k/instance | 95.2805 | 2.8337 | [91.7620, 98.7990] | 2.5345 |
| results_cv_en_shuffle.json | wideband_16k/none | 91.8020 | 3.8172 | [87.0624, 96.5417] | 3.4142 |
| results_cv_en_shuffle.json | wideband_16k/instance | 89.7158 | 4.3076 | [84.3672, 95.0644] | 3.8528 |
| results_cv_en_shuffle.json | bandrestrict_4k/none | 86.7991 | 8.3451 | [76.4373, 97.1610] | 7.4641 |
| results_cv_en_shuffle.json | bandrestrict_4k/instance | 85.6414 | 4.8909 | [79.5686, 91.7142] | 4.3745 |
| results_cv_en_shuffle.json | lowpass_4k/none | 91.2108 | 5.1238 | [84.8488, 97.5728] | 4.5829 |
| results_cv_en_shuffle.json | lowpass_4k/instance | 89.3885 | 4.9230 | [83.2758, 95.5011] | 4.4032 |
| results_cv_gn.json | wideband_16k/none | 58.2350 | 50.7891 | [-4.8280, 121.2981] | 45.4272 |
| results_cv_gn.json | wideband_16k/instance | 63.3420 | 46.6054 | [5.4738, 121.2102] | 41.6851 |
| results_cv_gn.json | bandrestrict_4k/none | 73.6332 | 39.6361 | [24.4184, 122.8479] | 35.4516 |
| results_cv_gn.json | bandrestrict_4k/instance | 74.6257 | 41.8634 | [22.6454, 126.6060] | 37.4438 |
| results_cv_gn.json | lowpass_4k/none | 58.7746 | 52.2050 | [-6.0465, 123.5956] | 46.6935 |
| results_cv_gn.json | lowpass_4k/instance | 63.1798 | 45.7505 | [6.3731, 119.9865] | 40.9205 |

## Model parameter counts

Instantiated from `experiments/lid/model.py` (untrained, architecture-only count of `requires_grad=True` parameters).

| Model | Trainable params | Approx. fp32 size (MiB) |
|---|---:|---:|
| CRNN_LID | 1,291,329 | 4.926 |
| CNN_LID | 101,441 | 0.387 |

Ratio CRNN/CNN = **12.73x**.

