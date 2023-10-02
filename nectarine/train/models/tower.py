from abc import abstractmethod

import tensorflow as tf


class Tower(tf.keras.Model):
    def __init__(
        self,
        n_dims: int,
        embedding_dim: int = 16,
        feature_layers: list[int] = None,
        output_layers: list[int] = None,
    ):
        super().__init__()
        self._n_dims = n_dims
        self._embedding_dim = embedding_dim
        self._feature_layers = feature_layers or [64, 32, 16]
        self._output_layers = output_layers or [64, 32]

        self.embedding_layer = tf.keras.layers.Embedding(n_dims + 1, embedding_dim)
        self.feature_model = self._create_model(layer_sizes=self._feature_layers)

    @abstractmethod
    def call(self, inputs, training: bool = None):
        raise NotImplementedError

    def get_config(self):
        return {
            "n_dims": self._n_dims,
            "embedding_dim": self._embedding_dim,
            "feature_layers": self._feature_layers,
            "output_layers": self._output_layers,
        }

    @staticmethod
    def _create_model(
        layer_sizes: list[int],
        layer: tf.keras.layers.Layer = tf.keras.layers.Dense,
        input_shape: tuple[int] = None,
        name: str = None,
        **kwargs,
    ):
        layers = [tf.keras.Input(shape=input_shape)] if input_shape else []
        layers += [layer(num_neurons, **kwargs) for num_neurons in layer_sizes]
        return tf.keras.Sequential(layers, name)


class CandidateModel(Tower):
    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 16,
        feature_layers: list[int] = None,
        output_layers: list[int] = None,
    ):
        super().__init__(n_items, embedding_dim, feature_layers, output_layers)
        self.output_model = self._create_model(
            self._output_layers,
            input_shape=(None, self._embedding_dim + self._feature_layers[-1]),
        )

    def call(self, inputs, *_):
        embedding = self.embedding_layer(inputs["id"])
        embedding = tf.squeeze(embedding)

        features = self.feature_model(inputs["features"])

        x = tf.concat([embedding, features], axis=1)
        x = self.output_model(x)
        return x


class QueryModel(Tower):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        feature_layers: list[int] = None,
        session_layers: list[int] = None,
        output_layers: list[int] = None,
    ):
        super().__init__(n_users, embedding_dim, feature_layers, output_layers)
        self._session_layers = session_layers or [32, 32]

        self.session_embedding_layer = tf.keras.layers.Embedding(
            input_dim=n_items,
            output_dim=embedding_dim,
        )

        self.session_model = self._create_model(
            layer=tf.keras.layers.LSTM,
            layer_sizes=self._session_layers[:-1],
            return_sequences=True,
        )
        session_output = tf.keras.layers.LSTM(
            self._session_layers[-1], return_sequences=False
        )
        self.session_model.add(session_output)

        output_dim = (
            self._embedding_dim + self._feature_layers[-1] + self._session_layers[-1]
        )
        self.output_model = self._create_model(
            layer_sizes=self._output_layers,
            input_shape=(None, output_dim),
        )

    def call(self, inputs, *_):
        embedding = self.embedding_layer(inputs["id"])
        embedding = tf.squeeze(embedding)

        features = self.feature_model(inputs["features"])

        sessions = tf.cast(inputs["session"], tf.int32)
        sessions = self.session_embedding_layer(sessions)
        sessions = self.session_model(sessions)

        x = tf.concat([embedding, features, sessions], axis=1)
        x = self.output_model(x)
        return x
