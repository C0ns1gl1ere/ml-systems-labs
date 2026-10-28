"""Importable transformers: saved joblib files can be loaded in a new process."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

NUMERIC = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'native-country']
DERIVED = ['capital_net', 'hours_per_age']


def normalize_missing(X):
    """Operate on a copy, after the split. OpenML may already decode '?' to NaN."""
    result = X.copy(deep=True)
    for column in CATEGORICAL:
        result[column] = result[column].astype(object).replace('?', np.nan)
    return result


class AdultCleaner(BaseEstimator, TransformerMixin):
    """Train-only imputation and quantile fitting; transform never deletes rows."""
    def __init__(self, quantiles=(0.01, 0.99), engineered=False):
        self.quantiles = quantiles
        self.engineered = engineered

    def fit(self, X, y=None):
        data = normalize_missing(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_fit_rows_ = len(X)
        self.numeric_imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        self.category_imputer_ = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
        numeric = self.numeric_imputer_.fit_transform(data[NUMERIC])
        self.category_imputer_.fit(data[CATEGORICAL])
        self.lower_ = np.quantile(numeric, self.quantiles[0], axis=0)
        self.upper_ = np.quantile(numeric, self.quantiles[1], axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, 'lower_')
        data = normalize_missing(X)
        result = pd.DataFrame(index=X.index)
        numeric = self.numeric_imputer_.transform(data[NUMERIC])
        result[NUMERIC] = np.clip(numeric, self.lower_, self.upper_)
        result[CATEGORICAL] = self.category_imputer_.transform(data[CATEGORICAL])
        result = result[list(self.feature_names_in_)]
        if self.engineered:
            # Formulas use the imputed, clipped base features, identically in both splits.
            result['capital_net'] = result['capital-gain'] - result['capital-loss']
            denominator = result['age'] - 16
            if (denominator <= 0).any():
                raise ValueError('hours_per_age requires age > 16')
            result['hours_per_age'] = result['hours-per-week'] / denominator
        return result
