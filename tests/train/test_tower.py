import tensorflow as tf

from nectarine.train.models.tower import CandidateModel, QueryModel


def test_query():
    inputs = tf.data.Dataset.from_tensor_slices(
        {
            "id": [[1], [2], [3]],
            "session": [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            "features": [[1.0, 0.8], [1.0, 0.4], [1.0, 0.2]],
        }
    )

    inputs = list(inputs.batch(3).take(1))[0]
    query_model = QueryModel(4, 5, 5)
    out = query_model(inputs)

    assert out.shape == [3, 32]


def test_candidate():
    inputs = tf.data.Dataset.from_tensor_slices(
        {
            "id": [[1], [2], [3]],
            "features": [[1.0, 0.8], [1.0, 0.4], [1.0, 0.2]],
        }
    )

    inputs = list(inputs.batch(3).take(1))[0]
    candidate_model = CandidateModel(3, 5)
    out = candidate_model(inputs)

    assert out.shape == [3, 32]
