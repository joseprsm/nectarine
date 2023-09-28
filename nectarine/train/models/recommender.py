import tensorflow as tf
import tensorflow_recommenders as tfrs

from .candidate import CandidateModel
from .query import QueryModel


class Recommender(tfrs.Model):
    """The main Recommender model.

    It expects a `tf.data.Dataset`, composed of two keys: "query" and "candidate";
    the query part of the dataset has three keys:

    * the user ID feature name, a scalar;
    * `user_features`, an array representing the user features
    * `context_features`, an array representing the context features

    The candidate part of the data set has two keys:

    * the item ID feature name, a scalar;
    * `item_features`, an array representing the item features

    The query tower model takes the user ID feature and passes it by an embedding layer. The
    user and context features are concatenated and passed by a number of dense layers. The
    item ID feature is similarly passed to an Embedding layer. Its outputs are then concatenated
    to the outputs of the features model whose inputs are the item features, and are then
    passed by a number of Dense layers.

    An optional Ranking model is also included, granted there are `ranking_features`.

    Args:
        user_dims (int): number possible values for the user ID feature
        item_dims (int): number possible values for the item ID feature
        embedding_dim (int): output dimension of the embedding layer
        feature_layers (list): number of neurons in each layer for the feature models
        output_layers (list): number of neurons in each layer for the output models

    Examples:
        >>> from rexify.models import Recommender
        >>> model = Recommender()
        >>> model.compile()

        >>> import numpy as np
        >>> inputs = tf.data.Dataset.from_tensor_slices(np.concatenate([np.random.randint(0, 15, size=100).reshape(-1, 1), np.random.randint(0, 1, size=100).reshape(-1, 1), np.random.randint(0, 1_000, size=100).reshape(-1, 1), np.random.randint(0, 1_000, size=100).reshape(-1, 1), np.random.randint(0, 15, size=100).reshape(-1, 1), np.random.randint(0, 5, size=100).reshape(-1, 1),], axis=1)).map(lambda x: {'query': {'user_id': x[0], 'user_features': x[1:3], 'context_features': x[3:4]}, 'candidate': {'item_id': x[4], 'item_features': x[5:]}}).batch(128)

        >>> _ = model.fit(inputs, verbose=0)

    """

    def __init__(self, config: dict):
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
        callbacks = callbacks or self._get_callbacks(x, batch_size)
        # todo: validate number of index callbacks
        #   - can't be more than a single index for each model (query, candidate)

        if batch_size:
            x = x.batch(batch_size)
            if validation_data:
                validation_data = validation_data.batch(batch_size)

        return super().fit(
            x, epochs=epochs, validation_data=validation_data, callbacks=callbacks
        )

    def get_config(self):
        return {
            "item_dims": self._item_dims,
            "user_dims": self._user_dims,
            "output_layers": self._output_layers,
            "feature_layers": self._feature_layers,
            "ranking_layers": self._ranking_layers,
            "ranking_features": self._ranking_features,
            "ranking_weights": self._ranking_weights,
        }

    @classmethod
    def load(cls, export_dir: str) -> tf.keras.Model:
        return tf.saved_model.load(export_dir)

    @staticmethod
    def _get_callbacks(x, batch_size: int = None) -> list[tf.keras.callbacks.Callback]:
        # required to set index shapes
        sample_query = ...

        def get_index_callback():
            try:
                import scann  # noqa: F401

                from nectarine.train.callbacks import ScaNNCallback

                return ScaNNCallback(sample_query, batch_size=batch_size)

            except ImportError:
                from nectarine.train.callbacks import BruteForceCallback

                return BruteForceCallback(sample_query, batch_size=batch_size)

        def get_mlflow_callback():
            try:
                from nectarine.train.callbacks import MlflowCallback

                return MlflowCallback()

            except ImportError:
                return

        callbacks = [get_index_callback(), get_mlflow_callback()]
        callbacks = callbacks[:-1] if callbacks[-1] is None else callbacks

        return callbacks
