import jax.numpy as jnp
import numpy as np
import pytest

from nectarine.transform.encode.features import CategoryEncoder, IDEncoder
from nectarine.transform.encode.min_max import MinMaxScaler
from nectarine.transform.encode.one_hot import OneHotEncoder


_one_hot_data = [
    (
        jnp.array([1, 2, 3, 2]).reshape(-1, 1),
        jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]),
    ),
]


@pytest.mark.parametrize(["x", "y"], _one_hot_data)
def test_one_hot(x, y):
    y_hat = OneHotEncoder().fit_transform(x)
    np.testing.assert_array_equal(y, y_hat)


_min_max_data = [
    (
        jnp.array([[1, 2], [2, 3], [3, 1]]),
        jnp.array([[0.0, 0.5], [0.5, 1.0], [1.0, 0.0]]),
    )
]


@pytest.mark.parametrize(["x", "y"], _min_max_data)
def test_min_max(x, y):
    y_hat = MinMaxScaler().fit_transform(x)
    np.testing.assert_array_almost_equal(y, y_hat)


_category_data = [(np.array([["a"], ["b"], ["a"]]), np.array([[1, 0], [0, 1], [1, 0]]))]


@pytest.mark.parametrize(["x", "y"], _category_data)
def test_category(x, y):
    y_hat = CategoryEncoder([None])._transformer.fit_transform(x)
    np.testing.assert_array_equal(y, y_hat)


_id_data = [(_category_data[0][0], np.array([[0], [1], [0]]))]


@pytest.mark.parametrize(["x", "y"], _id_data)
def test_id(x, y):
    y_hat = IDEncoder([None])._transformer.fit_transform(x)
    np.testing.assert_array_equal(y, y_hat)
