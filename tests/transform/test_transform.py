import string

import numpy as np
import pandas as pd
import pytest

from nectarine.transform import FeatureTransformer


data = pd.DataFrame(
    {
        "user_id": np.random.choice(list(string.ascii_lowercase), size=1000),
        "age": np.random.choice(100, size=1000),
        "gender": np.random.choice(["m", "f"], size=1000),
    }
)

schema = {"user_id": "id", "age": "number", "gender": "category"}


@pytest.mark.parametrize(["header"], [(["user_id", "age", "gender"],), (None,)])
def test_fit(header):
    feat = FeatureTransformer(schema, header=header)
    feat.fit(data)


@pytest.mark.parametrize(["header"], [(["user_id", "age", "gender"],), (None,)])
def test_transform(header):
    feat = FeatureTransformer(schema, header=header)
    transformed_data = feat.fit(data).transform(data)
    ids, feats = transformed_data
    assert ids.shape == (data.shape[0], 1)
    assert feats.shape == (data.shape[0], 3)


@pytest.mark.parametrize(["header"], [(["user_id", "age", "gender"],), (None,)])
def test_encode(header):
    feat = FeatureTransformer(schema, header).fit(data)
    encoded_col = feat.encode(data, return_dataframe=False)
    assert encoded_col.dtype == int
    encoded_df = feat.encode(data)
    assert encoded_df.shape == data.shape
