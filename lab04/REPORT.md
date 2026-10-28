# Lab 4 - Data pipeline and quality

## Problem definition

This experiment builds a reproducible binary income classifier using UCI Adult,
retrieved through scikit-learn's `fetch_openml(name='adult', version=2, as_frame=True)`.
The task is to predict whether annual income exceeds USD 50,000. The inputs are
14 census attributes: age, workclass, fnlwgt, education, education-num,
marital-status, occupation, relationship, race, sex, capital-gain, capital-loss,
hours-per-week and native-country. The output is income, with >50K encoded as 1
and <=50K as 0. This is a historical educational benchmark, not an estimate of
current income or a basis for decisions about individuals.

Success is measured using accuracy and binary F1 for the >50K class. Accuracy
measures overall correctness, while F1 balances precision and recall for the
positive class and is useful when the classes are imbalanced. A fixed 70/30
stratified split with seed 42 preserves the approximate class proportions.
Both model variants use exactly the same rows, fixed LogisticRegression settings
and the default decision threshold. No performance threshold is invented, and
the test results are not used to tune parameters or choose a production model.

The primary constraints are leakage prevention, traceability and reproducibility.
The split occurs before cleaning; only training rows determine medians, modes,
clipping thresholds, scaling and encoding. Raw inputs are preserved. Validation
failures are reported explicitly and stop execution. Every intentional removal
is logged. The fitted transformers, split indices, configuration, versions and
processed data hash are saved so that another run can be checked against this one.

## Ingestion and validation

- Raw size: 48,842 rows, 15 columns.
- OpenML's target column is named `class`; a separate view renames it `income`.
- Schema violations: 0. See `results/validation_report.json` for all 15 column definitions and checks.
- Numeric values must have integer semantics; OpenML's integral float storage is accepted.
- Checked lab ranges: age 17-90, hours-per-week 1-99, capital-gain 0-99999, capital-loss 0-4356.
- Additional dictionary checks: education-num 1-16, fnlwgt > 1, categorical membership.
- Raw exact duplicate excess rows: 52.
- `?` is an allowed raw feature token meaning missing. OpenML can already represent it as NA.
- Missing inputs are reported and imputed, not treated as rows to delete; a missing target is an error.

| Column with missing values | Missing rows |
| --- | ---: |
| workclass | 2799 |
| occupation | 2809 |
| native-country | 857 |

## Split, cleaning and leakage controls

- Train before duplicate removal: 34,189; train after: 34,165.
- Exact duplicate training records removed: 24 (features plus target).
- Test: 14,653 rows; no deletion or label edits.
- >50K fractions after training deduplication: train 0.2394, test 0.2393.
- All imputation, clipping (1st/99th percentile), scaling and one-hot fitting uses deduplicated train only.
- Train-derived transformations are applied to test; 'untouched' means the raw test records and membership are retained.
- Unknown test categories encode as all zeros; no category vocabulary is learned from test.
- Test rows whose raw feature vector also appears in train: 27.
  The required random row split does not guarantee separation of identical feature groups.
  This possible optimism is disclosed; a future grouped evaluation would be a separate experiment.
- Automated integrity checks: Raw unchanged, test unchanged, disjoint row IDs, train-only fit, no train duplicates, convergence: PASS.

## Feature engineering and baseline comparison

The engineered variant adds `capital_net = capital-gain - capital-loss` and
`hours_per_age = hours-per-week / (age - 16)` after imputation and clipping.
The validated age domain makes the denominator positive. Derived numerical
features are standardized using training statistics. Capital net is a linear
combination of existing inputs, so an improvement is not guaranteed.

| Variant | Test accuracy | Test F1 (>50K) | Encoded columns |
| --- | ---: | ---: | ---: |
| baseline | 0.853955 | 0.660102 | 105 |
| engineered | 0.857777 | 0.669626 | 107 |

F1 change (engineered minus baseline): +0.009524.
These are descriptive results from one fixed split, without a significance claim.
Confusion matrices (rows=true, columns=predicted, order <=50K, >50K) are in `results/metrics.json`.

## Label quality and manual review

The read-only full-data audit found **5 distinct feature groups**
with conflicting income labels, containing **10 rows**.
See `results/label_conflicts.csv`. Conflicts are reported, not automatically resolved.
The audit is diagnostic and does not guide model fitting or test removal.

A reproducible random sample of 50 raw rows was checked automatically
against dictionary ranges, categories and education/education-num consistency.
Automated flagged sample rows: 0.
**The required human review remains pending.** Open `results/audit_50_manual.csv`,
inspect each row against `DATA_DICTIONARY.md`, fill the review status and notes,
and retain the reviewed file. No human review or independent label verification
is claimed here. Cohen's kappa and a 24-hour relabelling exercise apply to the
alternative subjective-label task, not this Adult audit.

## Reproducibility and data documentation

Run `python run.py` after installing the pinned requirements, or `./run_lab.ps1`
on Windows to create a local environment, install dependencies and execute.
The first data fetch needs internet; subsequent runs use the included local snapshot.
The raw CSV and original OpenML frame are preserved; all processing uses copies.

Processed CSV SHA256:

`f666a689a2f89063cfad480a993ad40bacf0feb7900ad5d4a4bdf6671655034a`

The hash covers deterministic UTF-8 CSV bytes: LF line endings, fixed column and
row order, 17 significant digits, training rows followed by test rows, original
row IDs, split labels, 16 cleaned/derived inputs and income. The exact bytes are
saved as `results/processed_dataset.csv`. Canonical hashes of the actual encoded
matrices, labels, row IDs and feature names are also in `results/config_and_hashes.yaml`.
The hash excludes binary container timestamps. Repeatability is expected with
the same snapshot and pinned environment; cross-version numerical identity is not promised.

See `data_card.md` for provenance, license, size, bias and privacy considerations.

Sources: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult),
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

## Execution verification

Two complete runs using the same snapshot and pinned environment produced
identical processed CSV hashes, all four canonical encoded-data hashes and
exactly identical metrics. The existing human-review worksheet was preserved.
Independent checks passed for train-only imputation, clipping, scaling and
category vocabularies, saved-model reloads, unseen-category handling and
validation rejection of invalid ranges/categories. See
`results/verification.json` and `results/reproducibility_check.json`.
