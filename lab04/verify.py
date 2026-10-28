"""Focused integrity checks. Run after run.py: python verify.py"""
import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.model_selection import train_test_split
from run import ROOT, FEATURES, TARGET, csv_bytes, sha, validate, make_preprocessor, matrix_digest


def main():
    cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
    results = ROOT / 'results'
    manifest = yaml.safe_load((results / 'config_and_hashes.yaml').read_text())
    raw = joblib.load(ROOT / 'data/raw/adult_openml.joblib')
    df = raw.rename(columns={raw.columns[-1]: TARGET}).copy()
    assert sha(csv_bytes(raw)) == manifest['raw_csv_sha256']
    assert sha((results / 'processed_dataset.csv').read_bytes()) == manifest['hashes']['processed_dataset.csv']
    splits = json.loads((results / 'split_indices.json').read_text())
    train_ids, test_ids = train_test_split(df.index.to_numpy(), test_size=cfg['test_size'],
                                          stratify=df[TARGET], random_state=cfg['random_state'])
    assert train_ids.tolist() == splits['train_before_dedup']
    assert test_ids.tolist() == splits['test']
    train = df.loc[train_ids].drop_duplicates()
    test = df.loc[test_ids]
    assert train.index.tolist() == splits['train_after_dedup']
    assert set(train.index).isdisjoint(test.index)
    X_train, X_test = train[FEATURES], test[FEATURES]
    checks = ['raw/processed SHA256', 'repeatable stratified split and exact training deduplication']
    for name in ['baseline', 'engineered']:
        fitted = joblib.load(results / f'{name}_model.joblib')
        preprocessor = fitted.named_steps['preprocessor']
        before = copy.deepcopy(preprocessor.named_steps['clean'].__dict__)
        transformed = preprocessor.transform(X_test)
        stored = sparse.load_npz(results / f'{name}_test_encoded.npz')
        np.testing.assert_allclose(transformed.toarray(), stored.toarray(), rtol=0, atol=0)
        clean = preprocessor.named_steps['clean']
        assert clean.n_fit_rows_ == len(train)
        # Independent train medians/quantiles: test outliers cannot affect fitted state.
        from transformers import NUMERIC, CATEGORICAL, normalize_missing
        np.testing.assert_allclose(clean.numeric_imputer_.statistics_, X_train[NUMERIC].median().to_numpy())
        norm = normalize_missing(X_train)
        expected_modes = [norm[c].mode().iloc[0] for c in CATEGORICAL]
        assert clean.category_imputer_.statistics_.tolist() == expected_modes
        imputed = clean.numeric_imputer_.transform(X_train[NUMERIC])
        np.testing.assert_allclose(clean.lower_, np.quantile(imputed, .01, axis=0))
        np.testing.assert_allclose(clean.upper_, np.quantile(imputed, .99, axis=0))
        training_cleaned = clean.transform(X_train)
        encoded_numeric = NUMERIC + (['capital_net', 'hours_per_age'] if name == 'engineered' else [])
        scaler = preprocessor.named_steps['encode'].named_transformers_['numeric']
        np.testing.assert_allclose(scaler.mean_, training_cleaned[encoded_numeric].mean().to_numpy())
        fitted_categories = preprocessor.named_steps['encode'].named_transformers_['categorical'].categories_
        for column, categories in zip(CATEGORICAL, fitted_categories):
            assert set(categories) == set(training_cleaned[column].unique())
        extreme = X_test.copy()
        extreme['age'] = 10**8
        extreme['workclass'] = extreme['workclass'].astype(object)
        extreme['workclass'] = 'NEW_UNSEEN_CATEGORY'
        preprocessor.transform(extreme)
        for key in ['lower_', 'upper_']:
            np.testing.assert_array_equal(getattr(clean, key), before[key])
        assert clean.numeric_imputer_.statistics_.tolist() == before['numeric_imputer_'].statistics_.tolist()
        categories = preprocessor.named_steps['encode'].named_transformers_['categorical'].categories_
        assert 'NEW_UNSEEN_CATEGORY' not in categories[0]
        names = preprocessor.named_steps['encode'].get_feature_names_out()
        for split, data in [('train', train), ('test', test)]:
            m = sparse.load_npz(results / f'{name}_{split}_encoded.npz')
            assert matrix_digest(m, data[TARGET].eq('>50K').astype(int), data.index, names) == manifest['hashes'][f'{name}_{split}_encoded_contents']
        prediction = fitted.predict(X_test)
        from sklearn.metrics import accuracy_score, f1_score
        metrics = json.loads((results / 'metrics.json').read_text())[name]
        assert accuracy_score(test[TARGET].eq('>50K'), prediction) == metrics['accuracy']
        assert f1_score(test[TARGET].eq('>50K'), prediction) == metrics['f1']
    checks += ['fresh-process model reload and exact saved transforms', 'train-only medians/modes/quantiles/scaling/vocabulary',
               'unseen test categories and extreme values do not change fitted state', 'encoded data hashes and saved metrics']
    bad = df.copy()
    bad.loc[bad.index[0], 'age'] = 100
    assert not validate(bad, cfg)['valid']
    bad = df.copy()
    bad['workclass'] = bad['workclass'].astype(object)
    bad.loc[bad.index[0], 'workclass'] = 'INVALID'
    assert not validate(bad, cfg)['valid']
    checks += ['validator rejects impossible range and unknown category']
    (results / 'verification.json').write_text(json.dumps({'status': 'PASS', 'checks': checks}, indent=2)+'\n')
    print('PASS: ' + '; '.join(checks))


if __name__ == '__main__':
    main()
