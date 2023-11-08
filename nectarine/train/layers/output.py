import jax.numpy as jnp
from flax import linen as nn


class OutputModel(nn.Module):
    layer_sizes: list[int]

    @nn.compact
    def __call__(self, *args):
        x = jnp.concatenate(args, axis=1)
        x = nn.LayerNorm()(x)
        for layer in self.layer_sizes[:-1]:
            x = nn.Dense(layer)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        x = nn.Dense(self.layer_sizes[-1])(x)
        return x
