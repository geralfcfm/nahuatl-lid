# Nahuatl–Spanish Language Identification — Code & Results

Reproducibility repository for the MSc thesis _“Channel-Matched Evaluation of Deep
Learning for Low-Resource Nahuatl Language Identification: Diagnosing Recording-Channel
Confounds”_ (Geraldine Lomeli Ponce, BUAP FCFM).

## Contents

- `experiments/lid/` — CRNN and CNN models (`model.py`), audio pipeline with the three
  spectral-band conditions incl. the true low-pass biquad (`audio.py`), speaker-grouped
  GroupKFold split (`data.py`), diagnostic probes (`transforms.py`), training/evaluation.
- `experiments/` — Modal serverless-GPU app (`modal_app.py`), Common Voice + baseline
  data loaders (`cv_data.py`, `preprocess.py`), configuration matrix runner (`run_matrix.py`).
- `experiments/analysis/` — paired-fold statistics (t-based CIs, TOST) and model
  parameter counts (`paired_stats.py`, `param_counts.py`).
- `results/` — per-configuration accuracy JSONs (with per-fold values), best-config
  confusion matrices, per-fold figures, and the derived `paired_stats.md` / `param_counts.md`.
  This includes the additional diagnostics reported in the thesis: the paired
  CNN-vs-CRNN architecture run (`results_cv_en_paired.json`) and its seed-variance
  sweep (`results_cv_en_paired_seeds.json`); the nested speaker-disjoint held-out test
  (`results_cv_{en,es}_nested.json`); the time-shuffle control repeated on the Spanish
  and Rarámuri contrasts (`results_cv_{es,tar}_shuffle.json`, alongside the English
  `results_cv_en_shuffle.json`); and per-contrast speaker/chunk counts
  (`results_speaker_counts.json`, produced by the `counts` mode in `modal_app.py`).

## Data (not included here)

- Field Nahuatl corpus (**TonalliCorpus**): https://tonallicorpus.com/?page_id=303
- Mexican Spanish: CIEMPIESS-Light (Hugging Face `ciempiess/ciempiess_light`).
- Contrast languages: Mozilla Common Voice — `es`/`en`/`gn` via the `fsicoli/common_voice_22_0`
  mirror; `ncx`/`tar`/`sei`/`qxp` from local Common Voice downloads.
  Raw audio is **not** redistributed here; obtain it from the sources above under their terms.

## Notes

- The main results use a single training seed (42); the seed-variance sweep
  (`results_cv_en_paired_seeds.json`, seeds 42/1/2) quantifies run-to-run training
  non-determinism. See the thesis for the statistical treatment (paired-fold confidence
  intervals / TOST, treated as descriptive over dependent folds) and its limitations.
- Author's role was **audio technician** for the TonalliCorpus recordings; corpus design,
  annotation, and authorship are credited to the TonalliCorpus team.
