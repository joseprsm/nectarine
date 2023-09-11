import numpy as np
import pytest

from nectarine.transform import Transform


@pytest.mark.parametrize(
    ["users", "items", "user_ids", "item_ids"],
    [
        (
            np.random.randint(100, size=90).reshape(-1, 3),
            np.random.randint(100, size=90).reshape(-1, 3),
            np.random.randint(89, size=10),
            np.random.randint(89, size=10),
        )
    ],
)
def test_transform_output(users, items, user_ids, item_ids):
    Transform(users, items).apply({}, user_ids, item_ids)
