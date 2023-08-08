from flax import linen as nn
from jax import numpy as jnp


class TransformOutput(nn.Module):
    user_lookup: nn.Module
    item_lookup: nn.Module

    def __call__(self, user_id, item_id):
        user_features = jnp.concatenate([user_id, self.user_lookup(user_id)], axis=1)
        item_features = jnp.concatenate([item_id, self.item_lookup(item_id)], axis=1)
        return user_features, item_features
