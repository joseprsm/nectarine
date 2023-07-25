import jax
import jax.numpy as jnp
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoder(BaseEstimator, TransformerMixin):
    _uniques: list[jax.Array]

    def fit(self, X):
        self._uniques = [jnp.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        transformed = []
        for i in range(X.shape[1]):
            eye = jnp.eye(len(self._uniques[i]))
            idx = jnp.searchsorted(self._uniques[i], X[:, i])
            transformed.append(eye[idx])
        return jnp.concatenate(transformed, axis=1)
