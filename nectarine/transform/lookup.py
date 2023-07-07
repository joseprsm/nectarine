from flax import linen as nn
from jax import numpy as jnp


class LookupModule(nn.Module):
    data: jnp.ndarray

    def __call__(self, input_ids):
        return jnp.take(self.data, input_ids, axis=0)
