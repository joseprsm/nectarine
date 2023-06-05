import jax.numpy as jnp

from sklearn.base import BaseEstimator, TransformerMixin


class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=None):
        super().__init__()
        self.feature_range = feature_range or (0, 1)

    def fit(self, X, *_):
        data_min = jnp.min(X, axis=0)
        data_max = jnp.max(X, axis=0)

        diff = data_max - data_min
        self._scale = (self.feature_range[1] - self.feature_range[0]) / diff
        self._min = self.feature_range[0] - data_min * self._scale
        self._max = jnp.max(X, axis=0)
        return self

    def transform(self, X):
        X *= self._scale
        X += self._min
        return X
