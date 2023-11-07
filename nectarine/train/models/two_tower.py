import jax.numpy as jnp
from flax import linen as nn

from ..layers.tower import CandidateTower, QueryTower


features = dict[str, jnp.ndarray]


class TwoTower(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, user: features, item: features):
        user_embeddings = QueryTower(**self.config["query"])(user)
        item_embeddings = CandidateTower(**self.config["candidate"])(item)
        return user_embeddings, item_embeddings
