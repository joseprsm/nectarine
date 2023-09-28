import tensorflow as tf


class QueryModel(tf.keras.Model):
    def __init__(self, n_dims, embedding_dim):
        self.embedding_layer = tf.keras.layers.Embedding(n_dims, embedding_dim)
        self.feature_model = ...
        self.output_model = ...
