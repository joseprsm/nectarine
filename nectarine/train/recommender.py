from flax import linen as nn
from jax import numpy as jnp

from ..transform import Transform
from .tower import Tower


class Recommender(nn.Module):
    config: dict
    transform: Transform

    @nn.compact
    def __call__(self, user_id: jnp.ndarray, item_id: jnp.ndarray):
        user, item = self.transform(user_id, item_id)
        query_embeddings = Tower(**self.config["query"])(user)
        candidate_embeddings = Tower(**self.config["candidate"])(item)
        return query_embeddings, candidate_embeddings
