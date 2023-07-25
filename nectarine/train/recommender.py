from flax import linen as nn
from jax import numpy as jnp

from .tower import Tower


class Recommender(nn.Module):
    schema: dict
    config: dict

    def setup(self):
        self.query_model = Tower(**self.config["query"])
        self.candidate_model = Tower(**self.config["candidate"])

    def __call__(self, user: jnp.ndarray, item: jnp.ndarray):
        query_embeddings = self.query_model(**user)
        candidate_embeddings = self.candidate_model(**item)
        return query_embeddings, candidate_embeddings
