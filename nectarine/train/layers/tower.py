import jax.numpy as jnp
from flax import linen as nn


class _Tower(nn.Module):
    n_dims: int
    embedding_dim: int = 16
    feature_layers: list[int] = None
    output_layers: list[int] = None

    @nn.compact
    def feature_model(self, features, activation: str = None):
        x = features
        for layer in self.feature_layers:
            x = nn.Dense(layer)(x)
            if activation:
                x = getattr(nn, activation)(x)
        return x

    @nn.compact
    def output(self, *args):
        x = jnp.concatenate(args, axis=1)
        x = nn.LayerNorm()(x)
        for layer in self.output_layers[:-1]:
            x = nn.Dense(layer)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_layers[-1])(x)
        return x


class QueryTower(_Tower):
    @nn.compact
    def __call__(self, inputs: dict[str, jnp.ndarray], *_):
        x = nn.Embed(self.n_dims, self.embedding_dim)(inputs["id"])
        features = self.feature_model(inputs["features"], "leaky_relu")
        return self.output(x, features)


class CandidateTower(_Tower):
    @nn.compact
    def __call__(self, inputs: dict[str, jnp.ndarray], *_):
        x = nn.Embed(self.n_dims, self.embedding_dim)(inputs["id"])
        features = self.feature_model(inputs["features"], "leaky_relu")
        return self.output(x, features)
