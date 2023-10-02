import tensorflow as tf

from nectarine import Recommender


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
    },
)

config = {
    "query": {"n_users": 4, "n_items": 5, "embedding_dim": 5},
    "candidate": {"n_items": 3, "embedding_dim": 5},
}

model = Recommender(config=config)
model.compile()


def test_recommender_call():
    inputs_ = list(inputs.batch(3).take(1))[0]  # noqa: F823
    query_embeddings, candidate_embeddings = model(inputs_)

    assert query_embeddings.shape == [3, 32]
    assert candidate_embeddings.shape == [3, 32]


def test_recommender_fit():
    model.fit(inputs, batch_size=3, epochs=2)
