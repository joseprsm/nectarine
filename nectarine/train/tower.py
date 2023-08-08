from flax import linen as nn
from jax import numpy as jnp


class Tower(nn.Module):
    n_dims: int
    embedding_dim: int = 32
    layer_sizes: list[int] = None
    name: str = None

    @nn.compact
    def __call__(self, features):
        id_, features = features[:, [0]].astype(int), features[:, 1:]

        x = nn.Embed(
            num_embeddings=self.n_dims,
            embedding_init=nn.initializers.xavier_uniform(),
            features=self.embedding_dim,
        )(id_)

        x = x.reshape(-1, self.embedding_dim)
        x = jnp.concatenate([x, features], axis=1)

        layers = self.layer_sizes or [64, 32]
        for layer in layers[:-1]:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(layers[-1])(x)

        return x
