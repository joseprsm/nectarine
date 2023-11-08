import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from ..layers import CandidateTower, QueryTower


features = dict[str, jnp.ndarray]


class TwoTower(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, user: features, item: features):
        user_embeddings = QueryTower(**self.config["query"])(user)
        item_embeddings = CandidateTower(**self.config["candidate"])(item)
        return user_embeddings, item_embeddings

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
            query_embeddings, candidate_embeddings = tuple(map(jnp.squeeze, embeddings))
            scores = jnp.matmul(query_embeddings, jnp.transpose(candidate_embeddings))
            n_queries, n_candidates, *_ = scores.shape

            labels = jnp.eye(n_queries, n_candidates)
            loss = categorical_crossentropy(labels, scores)
            return loss

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(state.params)
        return grads, loss
