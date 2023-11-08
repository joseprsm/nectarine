import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from ..layers.tower import CandidateTower, QueryTower


features = dict[str, jnp.ndarray]


class TwoTower(nn.Module):
    n_users: int
    n_items: int
    embedding_dim: int = 16
    feature_layers: list[int] = None
    output_layers: list[int] = None

    def setup(self):
        self.query_tower = QueryTower(
            n_dims=self.n_users,
            embedding_dim=self.embedding_dim,
            feature_layers=self.feature_layers,
            output_layers=self.output_layers,
        )

        self.candidate_tower = CandidateTower(
            n_dims=self.n_items,
            embedding_dim=self.embedding_dim,
            feature_layers=self.feature_layers,
            output_layers=self.output_layers,
        )

    def __call__(self, user: features, item: features):
        query_embeddings = self.query_tower(user)
        candidate_embeddings = self.candidate_tower(item)
        return query_embeddings, candidate_embeddings

    @jax.jit
    @staticmethod
    def retrieval(state: TrainState, query: features, candidate: features):
        def categorical_crossentropy(y_true, y_pred):
            epsilon = 1e-8
            y_pred = jnp.clip(y_pred, a_min=-10, a_max=10)
            log_probs = jax.nn.log_softmax(y_pred, axis=-1)
            loss = -jnp.sum(y_true * (log_probs + epsilon))
            return jnp.mean(loss)

        def loss_fn(params):
            embeddings = state.apply_fn({"params": params}, query, candidate)
            embeddings = tuple(map(jnp.squeeze, embeddings))
            scores: jnp.ndarray = jnp.matmul(embeddings, jnp.transpose(embeddings[1]))
            n_queries, n_candidates, *_ = scores.shape

            labels = jnp.eye(n_queries, n_candidates)
            loss = categorical_crossentropy(labels, scores)
            return loss

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(state.params)
        return grads, loss
