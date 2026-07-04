# Nahuatl–Spanish Language Identification — Code & Results

Reproducibility repository for the MSc thesis *“Spectral Robustness in Low-Resource
Language Identification: A Comparative Deep Learning Study of Nahuatl and Mexican
Spanish”* (Geraldine Lomeli Ponce, BUAP FCFM).

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

## Data (not included here)
- Field Nahuatl corpus (**TonalliCorpus**): https://tonallicorpus.com/?page_id=303
- Mexican Spanish: CIEMPIESS-Light (Hugging Face `ciempiess/ciempiess_light`).
- Contrast languages: Mozilla Common Voice — `es`/`en`/`gn` via the `fsicoli/common_voice_22_0`
  mirror; `ncx`/`tar`/`sei`/`qxp` from local Common Voice downloads.
Raw audio is **not** redistributed here; obtain it from the sources above under their terms.

## Notes
- Results reflect a single training seed (42); see the thesis for the statistical treatment
  (paired-fold confidence intervals / TOST) and its limitations.
- Author's role was **audio technician** for the TonalliCorpus recordings; corpus design,
  annotation, and authorship are credited to the TonalliCorpus team.
