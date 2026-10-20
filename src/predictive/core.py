from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier

FEATURES = ['age_years', 'sex_0male_1female', 'episode_number']
TARGET = 'hospital_outcome_1alive_0dead'
FILES = {'primary': '1_primary_cohort.csv', 'study': '3_study_cohort.csv', 'korea': '2_validation_cohort.csv'}
CANDIDATES = ['prevalence', 'age_lr', 'all_lr', 'spline_lr', 'gradient_boosting']
ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_inputs(frame):
    if list(frame.columns) != FEATURES:
        raise ValueError('Expected age_years, sex_0male_1female, episode_number in this order.')
    x = frame.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
        raise ValueError('All inputs must be finite integers.')
    if not ((x[:, 0] >= 0) & (x[:, 0] <= 100)).all():
        raise ValueError('Age must be between 0 and 100 years.')
    if not np.isin(x[:, 1], [0, 1]).all():
        raise ValueError('Sex must be encoded as 0 or 1.')
    if not ((x[:, 2] >= 1) & (x[:, 2] <= 5)).all():
        raise ValueError('Episode number must be between 1 and 5.')
    return x

def load_data(cohort):
    path = ROOT / 'resources' / 'sepsis_survival_dataset' / FILES[cohort]
    df = pd.read_csv(path)
    if list(df.columns) != FEATURES + [TARGET] or df.isna().any().any():
        raise ValueError(f'Invalid schema or missing values: {path}')
    validate_inputs(df[FEATURES])
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError('Outcome must be encoded as 0 or 1.')
    return df, 1 - df[TARGET].to_numpy(dtype=int)

class Features(BaseEstimator, TransformerMixin):
    """Restricted cubic age spline with four training-fold quantile knots."""
    def __init__(self, kind='all_lr'):
        self.kind = kind

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float)
        if self.kind == 'spline_lr':
            self.knots_ = np.quantile(x[:, 0], [0.05, 0.35, 0.65, 0.95])
            if len(np.unique(self.knots_)) != 4:
                raise ValueError('Four distinct age knots required.')
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float)
        if self.kind == 'age_lr':
            return x[:, :1]
        if self.kind != 'spline_lr':
            return x
        age = x[:, 0]
        k = self.knots_
        cols = [age]
        for knot in k[:-2]:
            z = (np.maximum(age-knot, 0)**3
                 - np.maximum(age-k[-2], 0)**3*(k[-1]-knot)/(k[-1]-k[-2])
                 + np.maximum(age-k[-1], 0)**3*(k[-2]-knot)/(k[-1]-k[-2])) / (k[-1]-k[0])**2
            cols.append(z)
        return np.column_stack(cols + [x[:, 1], x[:, 2]])

def make_model(name, seed=20261012):
    if name == 'prevalence':
        return DummyClassifier(strategy='prior')
    if name == 'gradient_boosting':
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
            max_leaf_nodes=7, min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
    return make_pipeline(Features(name), StandardScaler(),
                         LogisticRegression(C=np.inf, max_iter=1000, tol=1e-8))

def predict_death(model, frame):
    validate_inputs(frame)
    index = list(model.classes_).index(1)
    return model.predict_proba(frame)[:, index]
