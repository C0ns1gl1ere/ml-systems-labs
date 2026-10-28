"""Lab 4: reproducible Adult data pipeline. Run: python run.py"""
import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from transformers import AdultCleaner, NUMERIC, CATEGORICAL, DERIVED

ROOT = Path(__file__).resolve().parent
TARGET = 'income'
FEATURES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country']
EDUCATION = dict(zip(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
                     '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm',
                     'Bachelors', 'Masters', 'Prof-school', 'Doctorate'], range(1, 17)))
CATEGORIES = {
    'workclass': 'Private|Self-emp-not-inc|Self-emp-inc|Federal-gov|Local-gov|State-gov|Without-pay|Never-worked',
    'education': '|'.join(EDUCATION),
    'marital-status': 'Married-civ-spouse|Divorced|Never-married|Separated|Widowed|Married-spouse-absent|Married-AF-spouse',
    'occupation': 'Tech-support|Craft-repair|Other-service|Sales|Exec-managerial|Prof-specialty|Handlers-cleaners|Machine-op-inspct|Adm-clerical|Farming-fishing|Transport-moving|Priv-house-serv|Protective-serv|Armed-Forces',
    'relationship': 'Wife|Own-child|Husband|Not-in-family|Other-relative|Unmarried',
    'race': 'White|Asian-Pac-Islander|Amer-Indian-Eskimo|Other|Black',
    'sex': 'Female|Male',
    'native-country': 'United-States|Cambodia|England|Puerto-Rico|Canada|Germany|Outlying-US(Guam-USVI-etc)|India|Japan|Greece|South|China|Cuba|Iran|Honduras|Philippines|Italy|Poland|Jamaica|Vietnam|Mexico|Portugal|Ireland|France|Dominican-Republic|Laos|Ecuador|Taiwan|Haiti|Columbia|Hungary|Guatemala|Nicaragua|Scotland|Thailand|Yugoslavia|El-Salvador|Trinadad&Tobago|Peru|Hong|Holand-Netherlands',
    'income': '<=50K|>50K',
}
RANGES = {'age': (17, 90), 'hours-per-week': (1, 99), 'capital-gain': (0, 99999),
          'capital-loss': (0, 4356), 'education-num': (1, 16)}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def csv_bytes(df, index=False):
    return df.to_csv(index=index, lineterminator='\n', float_format='%.17g', na_rep='?').encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ingest(cfg):
    if (cfg['dataset']['name'], cfg['dataset']['version']) != ('adult', 2):
        raise ValueError('This schema and raw cache are specific to adult version 2')
    raw_dir = ROOT / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / 'adult_openml.joblib'
    # This file is created locally by this project; do not load untrusted joblib files.
    if cache.exists():
        raw = joblib.load(cache)
    else:
        bunch = fetch_openml(name=cfg['dataset']['name'], version=cfg['dataset']['version'],
                             as_frame=True, data_home=str(ROOT / 'data' / 'openml_cache'))
        raw = bunch.frame.copy(deep=True)
        joblib.dump(raw, cache, compress=3)
    # Preserve the original OpenML column name ('class') in the immutable snapshot.
    snapshot = csv_bytes(raw)
    snapshot_path = raw_dir / 'adult_openml.csv'
    if snapshot_path.exists() and snapshot_path.read_bytes() != snapshot:
        raise ValueError('Raw snapshot does not match cache; refusing to overwrite it')
    if not snapshot_path.exists():
        snapshot_path.write_bytes(snapshot)
    logging.info('Ingestion: %s rows, %s columns; raw SHA256 %s', *raw.shape, sha(snapshot))
    view = raw.rename(columns={raw.columns[-1]: TARGET}).copy(deep=True)
    return raw, view, sha(snapshot)


def validate(df, cfg):
    expected = FEATURES + [TARGET]
    report = {'shape': list(df.shape), 'violations': [], 'columns': {},
              'null_policy': "Literal '?' and existing NA are missing; '?' is an allowed feature token. Target cannot be missing.",
              'type_policy': 'Integer semantics: numeric storage with finite whole-number values; categorical/object storage for categories.'}
    def issue(column, rule, count, rows=None):
        if count:
            report['violations'].append({'column': column, 'rule': rule, 'count': int(count),
                                          'example_row_ids': [] if rows is None else [int(i) for i in rows[:10]]})
    issue('*', 'expected 48842 rows', int(len(df) != cfg['dataset']['expected_rows']))
    issue('*', 'expected 15 columns', int(df.shape[1] != cfg['dataset']['expected_columns']))
    issue('*', 'column names and order', int(list(df.columns) != expected))
    for column in expected:
        if column not in df:
            continue
        s = df[column]
        missing = s.isna() | s.astype(object).eq('?')
        entry = {'expected_type': 'int' if column in NUMERIC else 'category', 'actual_dtype': str(s.dtype),
                 'native_na': int(s.isna().sum()), 'literal_question_marks': int(s.astype(object).eq('?').sum()),
                 'missing_total': int(missing.sum())}
        if column in NUMERIC:
            number = pd.to_numeric(s, errors='coerce')
            bad = ~missing & (number.isna() | ~np.isfinite(number) | (number % 1 != 0))
            issue(column, 'finite integer values', bad.sum(), df.index[bad])
            issue(column, 'numeric storage', int(not pd.api.types.is_numeric_dtype(s)))
            if column in RANGES:
                low, high = RANGES[column]
                bad = ~missing & ~number.between(low, high)
                entry['allowed_range'] = [low, high]
                issue(column, 'range', bad.sum(), df.index[bad])
            if column == 'fnlwgt':
                bad = ~missing & (number <= 1)
                issue(column, 'UCI extraction fnlwgt > 1', bad.sum(), df.index[bad])
        else:
            allowed = CATEGORIES[column].split('|') + ([] if column == TARGET else ['?'])
            bad = ~missing & ~s.isin(allowed)
            entry['allowed_categories'] = allowed
            issue(column, 'unknown category', bad.sum(), df.index[bad])
            category_storage = isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            issue(column, 'categorical storage', int(not category_storage))
        if column == TARGET:
            issue(column, 'missing target', missing.sum(), df.index[missing])
        report['columns'][column] = entry
    report['exact_duplicate_excess_rows'] = int(df.duplicated().sum())
    report['valid'] = not report['violations']
    return report


def label_audit(df, cfg, results):
    # Read-only diagnostic on raw values: not used to fit preprocessing or remove test rows.
    grouped = df.groupby(FEATURES, observed=True, dropna=False)[TARGET].agg(['nunique', 'size'])
    conflicts = grouped[grouped['nunique'] > 1].reset_index()
    conflicts.to_csv(results / 'label_conflicts.csv', index=False, lineterminator='\n')
    sample = df.sample(n=cfg['audit_sample_size'], random_state=cfg['random_state']).copy()
    sample.insert(0, 'source_row_id', sample.index)
    checks = []
    for _, row in sample.iterrows():
        notes = []
        for col, (low, high) in RANGES.items():
            if pd.notna(row[col]) and not low <= row[col] <= high:
                notes.append(f'{col} outside dictionary range')
        for col, values in CATEGORIES.items():
            if pd.notna(row[col]) and row[col] != '?' and row[col] not in values.split('|'):
                notes.append(f'{col} unknown category')
        if pd.notna(row['education']) and EDUCATION.get(row['education']) != row['education-num']:
            notes.append('education/education-num mismatch')
        if pd.notna(row['fnlwgt']) and row['fnlwgt'] <= 1:
            notes.append('fnlwgt <= 1')
        checks.append('; '.join(notes) or 'No impossible values found by automated checks')
    sample['automated_dictionary_check'] = checks
    sample['human_review_status'] = 'PENDING'
    sample['human_notes'] = ''
    # Never overwrite a user's existing manual-review worksheet.
    sample.to_csv(results / 'audit_50_auto.csv', index=False, lineterminator='\n')
    human = results / 'audit_50_manual.csv'
    if not human.exists():
        sample.to_csv(human, index=False, lineterminator='\n')
    return {'conflicting_feature_groups': len(conflicts),
            'rows_in_conflicting_groups': int(conflicts['size'].sum()),
            'sample_size': len(sample),
            'automated_sample_flags': sum(x != 'No impossible values found by automated checks' for x in checks),
            'manual_review_status': 'Not certified by pipeline; inspect and fill audit_50_manual.csv. Source labels cannot be independently verified.'}


def make_preprocessor(cfg, engineered):
    numeric = NUMERIC + (DERIVED if engineered else [])
    encoder = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CATEGORICAL),
    ], sparse_threshold=1.0)
    return Pipeline([('clean', AdultCleaner(tuple(cfg['clip_quantiles']), engineered)), ('encode', encoder)])


def matrix_digest(matrix, y, row_ids, names):
    # Stable canonical contents, independent of ZIP timestamps in scipy .npz containers.
    matrix = sparse.csr_matrix(matrix).copy()
    matrix.sum_duplicates()
    matrix.sort_indices()
    h = hashlib.sha256()
    for item in [np.asarray(matrix.shape, dtype='<i8'), matrix.indptr.astype('<i8'),
                 matrix.indices.astype('<i8'), matrix.data.astype('<f8'),
                 np.asarray(y, dtype='<i8'), np.asarray(row_ids, dtype='<i8')]:
        h.update(item.tobytes(order='C'))
    h.update(json.dumps(list(names), ensure_ascii=True, separators=(',', ':')).encode())
    return h.hexdigest()


def main(config_path=ROOT / 'config.yaml'):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.StreamHandler(), logging.FileHandler(results / 'run.log', mode='w', encoding='utf-8')], force=True)
    raw, df, raw_hash = ingest(cfg)
    raw_before = raw.copy(deep=True)
    validation = validate(df, cfg)
    write_json(results / 'validation_report.json', validation)
    if not validation['valid']:
        raise ValueError('Schema violations; see validation_report.json. No rows silently removed.')
    # The first modelling operation is the stratified split. No cleaning has occurred.
    train_ids, test_ids = train_test_split(df.index.to_numpy(), test_size=cfg['test_size'],
                                          stratify=df[TARGET], random_state=cfg['random_state'])
    train = df.loc[train_ids].copy(deep=True)
    test = df.loc[test_ids].copy(deep=True)
    test_before = test.copy(deep=True)
    original_train_size = len(train)
    duplicates = train.duplicated(keep='first')  # All 14 inputs AND target: exact records.
    pd.DataFrame({'source_row_id': train.index[duplicates]}).to_csv(results / 'train_duplicates_removed.csv', index=False)
    train = train.loc[~duplicates].copy()
    y_train = train[TARGET].eq('>50K').astype(int)
    y_test = test[TARGET].eq('>50K').astype(int)
    X_train, X_test = train[FEATURES], test[FEATURES]
    audit = label_audit(df, cfg, results)
    write_json(results / 'label_quality_report.json', audit)
    write_json(results / 'split_indices.json', {'train_before_dedup': train_ids.tolist(),
               'train_after_dedup': train.index.tolist(), 'test': test.index.tolist()})
    train_keys = pd.MultiIndex.from_frame(train[FEATURES])
    test_keys = pd.MultiIndex.from_frame(test[FEATURES])
    overlap_rows = int(test_keys.isin(train_keys).sum())
    metrics, hashes = {}, {}
    for name, engineered in [('baseline', False), ('engineered', True)]:
        preprocessor = make_preprocessor(cfg, engineered)
        train_matrix = preprocessor.fit_transform(X_train)
        test_matrix = preprocessor.transform(X_test)
        model = LogisticRegression(**cfg['model'], random_state=cfg['random_state'])
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train_matrix, y_train)
        prediction = model.predict(test_matrix)
        metrics[name] = {'accuracy': float(accuracy_score(y_test, prediction)),
                         'f1': float(f1_score(y_test, prediction, pos_label=1, zero_division=0)),
                         'confusion_matrix': confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
                         'encoded_features': train_matrix.shape[1], 'iterations': int(model.n_iter_[0])}
        # Whole pipeline accepts original, uncleaned feature rows at inference time.
        fitted = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        joblib.dump(fitted, results / f'{name}_model.joblib', compress=3)
        joblib.dump(preprocessor, results / ('preprocessor.joblib' if engineered else 'baseline_preprocessor.joblib'), compress=3)
        clean = preprocessor.named_steps['clean']
        write_json(results / f'{name}_fitted_statistics.json', {
            'n_fit_rows': clean.n_fit_rows_, 'numeric_columns': NUMERIC,
            'medians': clean.numeric_imputer_.statistics_.tolist(),
            'category_columns': CATEGORICAL, 'modes': clean.category_imputer_.statistics_.tolist(),
            'clip_lower': clean.lower_.tolist(), 'clip_upper': clean.upper_.tolist()})
        names = preprocessor.named_steps['encode'].get_feature_names_out().tolist()
        write_json(results / f'{name}_feature_names.json', names)
        for split, data, matrix, labels in [('train', X_train, train_matrix, y_train), ('test', X_test, test_matrix, y_test)]:
            matrix = sparse.csr_matrix(matrix)
            sparse.save_npz(results / f'{name}_{split}_encoded.npz', matrix)
            hashes[f'{name}_{split}_encoded_contents'] = matrix_digest(matrix, labels, data.index, names)
        if engineered:
            processed = []
            for split, data, labels in [('train', X_train, y_train), ('test', X_test, y_test)]:
                part = clean.transform(data)
                part.insert(0, 'source_row_id', data.index)
                part.insert(1, 'split', split)
                part['income'] = np.where(labels.to_numpy() == 1, '>50K', '<=50K')
                processed.append(part)
            content = csv_bytes(pd.concat(processed, ignore_index=True))
            (results / 'processed_dataset.csv').write_bytes(content)
            hashes['processed_dataset.csv'] = sha(content)
        np.testing.assert_array_equal(fitted.predict(X_test), prediction)
        assert clean.n_fit_rows_ == len(train)
        logging.info('%s: accuracy=%.6f F1=%.6f', name, metrics[name]['accuracy'], metrics[name]['f1'])
    pd.testing.assert_frame_equal(raw, raw_before)
    pd.testing.assert_frame_equal(test, test_before)
    assert set(train_ids).isdisjoint(set(test_ids))
    assert len(test) == int(np.ceil(len(df) * cfg['test_size']))
    assert not train.duplicated().any()
    write_json(results / 'metrics.json', metrics)
    environment = {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in
                   ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'PyYAML', 'matplotlib']}}
    manifest = {'configuration': cfg, 'raw_csv_sha256': raw_hash, 'hashes': hashes,
                'environment': environment, 'train_rows_before_dedup': original_train_size,
                'train_rows': len(train), 'test_rows': len(test),
                'train_duplicates_removed': original_train_size-len(train),
                'test_rows_with_feature_match_in_train': overlap_rows,
                'positive_train_fraction': float(y_train.mean()), 'positive_test_fraction': float(y_test.mean()),
                'automatic_checks': 'Raw unchanged, test unchanged, disjoint row IDs, train-only fit, no train duplicates, convergence: PASS'}
    (results / 'config_and_hashes.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')
    (results / 'processed_dataset.sha256').write_text(hashes['processed_dataset.csv']+'  processed_dataset.csv\n', encoding='ascii')
    generate_report(validation, audit, metrics, manifest)
    plot_metrics(metrics, results)
    logging.info('Done. Report: %s', ROOT / 'REPORT.md')
    return manifest


def plot_metrics(metrics, results):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), layout='constrained')
    for ax, metric in zip(axes, ['accuracy', 'f1']):
        bars = ax.bar(['Baseline', '+ 2 features'], [metrics[k][metric] for k in metrics], color=['#416b96', '#24958b'])
        ax.set_ylim(0, 1)
        ax.set_title('Accuracy' if metric == 'accuracy' else 'F1 (>50K)')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02, f'{bar.get_height():.4f}', ha='center')
    fig.suptitle('Adult: fixed stratified 30% test set')
    fig.savefig(results / 'metrics.png', dpi=160)
    plt.close(fig)


def generate_report(validation, audit, metrics, manifest):
    missing = '\n'.join(f"| {c} | {v['missing_total']} |" for c, v in validation['columns'].items() if v['missing_total'])
    scores = '\n'.join(f"| {k} | {v['accuracy']:.6f} | {v['f1']:.6f} | {v['encoded_features']} |" for k, v in metrics.items())
    text = f"""# Lab 4 - Data pipeline and quality

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

- Raw size: {validation['shape'][0]:,} rows, {validation['shape'][1]} columns.
- OpenML's target column is named `class`; a separate view renames it `income`.
- Schema violations: {len(validation['violations'])}. See `results/validation_report.json` for all 15 column definitions and checks.
- Numeric values must have integer semantics; OpenML's integral float storage is accepted.
- Checked lab ranges: age 17-90, hours-per-week 1-99, capital-gain 0-99999, capital-loss 0-4356.
- Additional dictionary checks: education-num 1-16, fnlwgt > 1, categorical membership.
- Raw exact duplicate excess rows: {validation['exact_duplicate_excess_rows']}.
- `?` is an allowed raw feature token meaning missing. OpenML can already represent it as NA.
- Missing inputs are reported and imputed, not treated as rows to delete; a missing target is an error.

| Column with missing values | Missing rows |
| --- | ---: |
{missing}

## Split, cleaning and leakage controls

- Train before duplicate removal: {manifest['train_rows_before_dedup']:,}; train after: {manifest['train_rows']:,}.
- Exact duplicate training records removed: {manifest['train_duplicates_removed']} (features plus target).
- Test: {manifest['test_rows']:,} rows; no deletion or label edits.
- >50K fractions after training deduplication: train {manifest['positive_train_fraction']:.4f}, test {manifest['positive_test_fraction']:.4f}.
- All imputation, clipping (1st/99th percentile), scaling and one-hot fitting uses deduplicated train only.
- Train-derived transformations are applied to test; 'untouched' means the raw test records and membership are retained.
- Unknown test categories encode as all zeros; no category vocabulary is learned from test.
- Test rows whose raw feature vector also appears in train: {manifest['test_rows_with_feature_match_in_train']}.
  The required random row split does not guarantee separation of identical feature groups.
  This possible optimism is disclosed; a future grouped evaluation would be a separate experiment.
- Automated integrity checks: {manifest['automatic_checks']}.

## Feature engineering and baseline comparison

The engineered variant adds `capital_net = capital-gain - capital-loss` and
`hours_per_age = hours-per-week / (age - 16)` after imputation and clipping.
The validated age domain makes the denominator positive. Derived numerical
features are standardized using training statistics. Capital net is a linear
combination of existing inputs, so an improvement is not guaranteed.

| Variant | Test accuracy | Test F1 (>50K) | Encoded columns |
| --- | ---: | ---: | ---: |
{scores}

F1 change (engineered minus baseline): {metrics['engineered']['f1']-metrics['baseline']['f1']:+.6f}.
These are descriptive results from one fixed split, without a significance claim.
Confusion matrices (rows=true, columns=predicted, order <=50K, >50K) are in `results/metrics.json`.

## Label quality and manual review

The read-only full-data audit found **{audit['conflicting_feature_groups']} distinct feature groups**
with conflicting income labels, containing **{audit['rows_in_conflicting_groups']} rows**.
See `results/label_conflicts.csv`. Conflicts are reported, not automatically resolved.
The audit is diagnostic and does not guide model fitting or test removal.

A reproducible random sample of {audit['sample_size']} raw rows was checked automatically
against dictionary ranges, categories and education/education-num consistency.
Automated flagged sample rows: {audit['automated_sample_flags']}.
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

`{manifest['hashes']['processed_dataset.csv']}`

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
"""
    (ROOT / 'REPORT.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    args = parser.parse_args()
    main(args.config)
