import tensorflow as tf

from nectarine import Recommender


def test_recommender():
    inputs = tf.data.Dataset.from_tensor_slices(
        {
            "query": {
                "id": [[1], [2], [3]],
                "session": [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                "features": [[1.0, 0.8], [1.0, 0.4], [1.0, 0.2]],
            },
            "candidate": {
                "id": [[1], [2], [3]],
                "features": [[1.0, 0.8], [1.0, 0.4], [1.0, 0.2]],
            },
        }
    )
    inputs = list(inputs.batch(3).take(1))[0]

    config = {
        "query": {"n_users": 4, "n_items": 5, "embedding_dim": 5},
        "candidate": {"n_items": 3, "embedding_dim": 5},
    }

    model = Recommender(config=config)
    query_embeddings, candidate_embeddings = model(inputs)

    assert query_embeddings.shape == [3, 32]
    assert candidate_embeddings.shape == [3, 32]
