import jax.numpy as jnp
from sklearn.base import BaseEstimator, TransformerMixin


class MinMaxScaler(BaseEstimator, TransformerMixin):
    _min: int
    _max: int

    def __init__(self, feature_range=None):
        self.feature_range = feature_range or (0, 1)

    def fit(self, X, *_):
        self._min = jnp.min(X, axis=0)
        self._max = jnp.max(X, axis=0)
        return self

    def transform(self, X):
        X = (X - self._min) / (self._max - self._min)
        return X
