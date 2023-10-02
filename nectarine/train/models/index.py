import tensorflow as tf
import tensorflow_recommenders as tfrs


class _BaseIndex:
    def __init__(self, query_model: tf.keras.Model):
        self.query_model = query_model

    def call(self, queries: tf.Tensor, k: int = None):
        inputs = (
            {"user_id": queries}
            if self.query_model.name.startswith("query")
            else {"item_id": queries}
        )
        return self.__class__.__bases__[1].call(self, inputs, k)


class BruteForce(_BaseIndex, tfrs.layers.factorized_top_k.BruteForce):
    def __init__(
        self,
        query_model: tf.keras.Model,
        window_size: int,
        k: int = 2,
        name: str = None,
    ):
        tfrs.layers.factorized_top_k.BruteForce.__init__(self, query_model, k, name)
        _BaseIndex.__init__(self, query_model, window_size)


class ScaNN(_BaseIndex, tfrs.layers.factorized_top_k.ScaNN):
    def __init__(
        self, query_model: tf.keras.Model, k: int = 10, name: str = None, **kwargs
    ):
        tfrs.layers.factorized_top_k.ScaNN.__init__(
            self, query_model, k, name, **kwargs
        )
        _BaseIndex.__init__(self, query_model)
