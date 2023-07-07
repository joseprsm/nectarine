from typing import Any

from flax import linen as nn


class Tower(nn.Module):
    n_dims: int
    embedding_dim: int = 32
    layer_sizes: list[int] = None
    name: str = None

    @nn.compact
    def __call__(self, id_, *_) -> Any:
        x = nn.Embed(
            num_embeddings=self.n_dims,
            embedding_init=nn.initializers.xavier_uniform(),
            features=self.embedding_dim,
        )(id_)

        layers = self.layer_sizes or [64, 32]
        for layer in layers[:-1]:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(layers[-1])(x)

        return x
