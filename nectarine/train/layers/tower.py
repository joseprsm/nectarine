import jax.numpy as jnp
from flax import linen as nn

from .features import FeaturesModel
from .output import OutputModel


class _Tower(nn.Module):
    n_dims: int
    embedding_dim: int
    feature_layers: list[int]
    output_layers: list[int]


class QueryTower(_Tower):
    @nn.compact
    def __call__(self, inputs: dict[str, jnp.ndarray], *_):
        x = nn.Embed(self.n_dims, self.embedding_dim)(inputs["id"])
        features = FeaturesModel(self.feature_layers, "leaky_relu")(inputs["features"])
        x = OutputModel(self.output_layers)(x, features)
        return x


class CandidateTower(_Tower):
    @nn.compact
    def __call__(self, inputs: dict[str, jnp.ndarray], *_):
        x = nn.Embed(self.n_dims, self.embedding_dim)(inputs["id"])
        features = FeaturesModel(self.feature_layers, "leaky_relu")(inputs["features"])
        x = OutputModel(self.output_layers)(x, features)
        return x
