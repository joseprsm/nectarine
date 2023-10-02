import tensorflow as tf
import tensorflow_recommenders as tfrs

from .tower import CandidateModel, QueryModel


class Recommender(tfrs.Model):
    """The main Recommender model.

    It expects a `tf.data.Dataset`, composed of two keys: "query" and "candidate";
    the query part of the dataset has three keys:

    * `id`, a scalar;
    * `features`, an array representing the user features
    * `session`, an array representing the previously interacted items
    * `context`, an array representing the context features

    The candidate part of the data set has two keys:

    * `id`, a scalar;
    * `features`, an array representing the item features

    The query tower model takes the user ID feature and passes it by an embedding layer. The
    user and context features are concatenated and passed by a number of dense layers. The
    item ID feature is similarly passed to an Embedding layer. Its outputs are then concatenated
    to the outputs of the features model whose inputs are the item features, and are then
    passed by a number of Dense layers.

    Args:
        config (dict): number possible values for the user ID feature

    Examples:
        >>> import tensorflow as tf
        >>> inputs = tf.data.Dataset.from_tensor_slices(
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
        >>> config = {
                "query": {"n_users": 4, "n_items": 5, "embedding_dim": 5},
                "candidate": {"n_items": 3, "embedding_dim": 5},
            }
        >>> from nectarine import Recommender
        >>> model = Recommender(config=config)
        >>> model.compile()
        >>> model.fit(inputs, batch_size=3, epochs=2, verbose=0)

    """

    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self.query_model = QueryModel(**config["query"])
        self.candidate_model = CandidateModel(**config["candidate"])
        self.retrieval_task = tfrs.tasks.Retrieval()

    def call(self, inputs, training=False):
        return (
            self.query_model(inputs["query"], training=training),
            self.candidate_model(inputs["candidate"], training=training),
        )

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        embeddings = self(inputs, training=training)
        loss = self.retrieval_task(*embeddings)
        return loss

    def fit(
        self,
        x: tf.data.Dataset,
        batch_size: int = None,
        epochs: int = 1,
        callbacks: list[tf.keras.callbacks.Callback] = None,
        validation_data=None,
    ):
        if batch_size:
            x = x.batch(batch_size)
            if validation_data:
                validation_data = validation_data.batch(batch_size)

        return super().fit(
            x, epochs=epochs, validation_data=validation_data, callbacks=callbacks
        )

    def get_config(self):
        return {"config": self._config}

    @classmethod
    def load(cls, export_dir: str) -> tf.keras.Model:
        return tf.saved_model.load(export_dir)
