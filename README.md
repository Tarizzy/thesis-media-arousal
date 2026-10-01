# Emotional intensity and factual reliability in political news

Code for the Master's thesis *[thesis title]* (Tarik Behmen, 2026).

The thesis asks whether the emotional intensity of political news writing is a stable property of the outlet
that publishes it, a response to the topic being covered, or neither, and how it relates to the outlet's factual
reliability. It analyses 57,210 political articles from 211 outlets (2017–2022), with Media Bias/Fact Check
factuality and bias ratings, using three associational arms: a cross-classified multilevel model (arousal as
outcome), an outlet-level ordered logit (factuality as outcome), and outlet-grouped nested cross-validation with
SHAP. A three-coder human validation checks the arousal measure.

## Data — not included

| Input | Where to get it | Place it at |
|---|---|---|
| misinfo-general corpus and `metadata.db` (Verhoeven et al., *Computational Linguistics* 52(2), 2026) | Hugging Face: `ioverho/misinfo-general` (CC BY-NC-SA 4.0) | `data/misinfo-general/` |
| NRC-VAD Lexicon v2.1 (Mohammad 2025) | saifmohammad.com/WebPages/nrc-vad.html | `NRC-VAD-Lexicon-v2.1/` |
| `final_sample.xlsx` (outlet selection) | built by `build_final_sample.py`; Ad Fontes Media lean scores were then added, which are not redistributed | project root |

Article text, article-level tables and the rating workbooks are not included: the corpus licence does not permit
redistribution, and the workbooks contain article excerpts. `validation/ratings_anonymised.csv` and
`validation/topic_ratings_anonymised.csv` contain the human ratings without names, text or article identifiers.

## Setup

```bash
pip install -r requirements.txt        # Python 3.13; full lock of the environment used
export THESIS_ROOT=/path/to/this/repo  # every script reads the project root from here
```

R 4.5.1 with `lme4` 2.0.6 and `lmerTest` 3.2.1 (`parallel` is base R). See `R_requirements.md` and
`r_sessioninfo.txt`. Random seed: 42 throughout (`pipeline/preprocessing.py`), with per-run offsets documented in
the scripts.

## Run order

Numbered prefixes give the order within each folder. Report files (`*_report.txt`, `*_checks.txt`) are the
outputs the thesis quotes and are included as a record.

**0 · Outlet selection** — `explore_db.py`, `explore_sources.py` (inspect the database); `build_manifest.py`
(source manifest); `build_final_sample.py` (eligible outlets → `final_sample.xlsx`); `check_rows.py`.

**1 · Pipeline** (`pipeline/`)
- `preprocessing.py` — shared tokenisation, political filter and feature definitions (imported by most scripts)
- `00_verify_join.py`, `00b_checks.py`, `00c_lexicon.py`, `00c_metadata_check.py`, `00d_checks.py` — input checks
- `02_keywords_v2.py`, `02a_validation_sheet.py`, `02b_text_artifacts.py`, `02c_filter_validation.py` — political
  headline filter: build, hand-validation sheet, text-artefact scan, precision/recall
- `sample_articles.py` — draw the analysis samples (up to 300 articles per outlet, year-stratified)
- `extract_features.py` — the eleven text features; articles need ≥ 50 matched content words
- `04_analysis_table.py` — join features, outlet ratings and topic clusters
- `00d_verify_rebuild.py`, `00e_verify_final.py` — verify the country fix and the random-stream fix
- `00g_descriptives.py` — descriptive statistics for the Data chapter
- `check_outputs.py`, `test_preprocessing.py`, `smoke_mixedlm.py` — tests and smoke checks

**2 · Topics** (`clustering/`) — `01_embed.py` (embed topic representations), `02_cluster.py` (K sweep, K = 33,
assignment; `k_sweep.png`), `03_cluster_coverage.py`. Labels and CAP codes: `cluster_labels.csv`.

**3 · Topic partisanship** (`partisanship/`) — decision rules fixed in advance: `05_expectations.md`.
`05a_counts.py` → `05b_partisanship.py` (coverage asymmetry; failed its reliability rule) → `05c_weighting.py` →
`05d_words.py` → `05e_framing.py` (framing divergence, the topic variable used) → `05f_compare_min500.py`
(500-token sensitivity).

**4 · Arm 1 and Arm 2** (`models/`) — `06a_model_frame.py` → `06b_mixed.R` (M0–M6) → `06c_compare_rerun.py` →
`07_outlet_ordlogit.py` (ordered logit S1–S4) → `07b_outliers.py` → `09a_frames.py` → `10_robustness.R`
(ten specifications, outlet bootstrap, leave-one-out) → `10b_no2018.R`.

**5 · Arm 3** (`ml/`) — decision rules: `08_expectations.md`. `08a_cv_compare.py` (leakage comparison) →
`08b2_nested_full.py` (nested CV; long run) → `08c_diagnostics.py` → `08d_explain.py` (SHAP, permutation
importance) → `08e_tfidf.py` (word-level benchmark).

**6 · Human validation** (`validation/`) — reading rule fixed before the second and third coders' ratings were
seen: `09_interpretation_rule.md`. `09_rating_sheet.py` (blind sheet) → `09b_rating_analysis.py` →
`09c_what_humans_saw.py` → `09d_three_coders.py` (three-coder analysis; authoritative for agreement).

Superseded scripts are kept in `archive/superseded/` with a note on what replaced each.

`pipeline/00b_checks.py` is a machine-specific diagnostic that searches the home folder; it is kept for the
record and is not needed to reproduce results.

## Licence

Code: MIT (see `LICENSE`). The corpus, the NRC-VAD lexicon, MBFC ratings and Ad Fontes Media scores are subject
to their own terms.
