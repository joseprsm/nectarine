from flax import linen as nn
from jax import numpy as jnp

from ..transform import TransformOutput
from .tower import Tower


class Recommender(nn.Module):
    schema: dict
    config: dict
    transform: TransformOutput

    def setup(self):
        self.query_model = Tower(**self.config["query"])
        self.candidate_model = Tower(**self.config["candidate"])

    def __call__(self, user_id: jnp.ndarray, item_id: jnp.ndarray):
        user, item = self.transform(user_id, item_id)
        query_embeddings = self.query_model(**user)
        candidate_embeddings = self.candidate_model(**item)
        return query_embeddings, candidate_embeddings
