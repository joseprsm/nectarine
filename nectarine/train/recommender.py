import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
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

    def fit(
        self,
        x: jnp.ndarray,
        batch_size: int,
        num_epochs: int = 1,
        rng=jax.random.PRNGKey(0),
    ):
        rng, init_rng = jax.random.split(rng)
        state = self._train_state(init_rng)
        for epoch in range(1, num_epochs + 1):
            state, train_loss = self._train_epoch(
                state, x, batch_size=batch_size, rng=rng
            )
            print(f"epoch: {epoch}, train_loss: {train_loss}")
        return state

    def _train_epoch(self, state, x, batch_size: int, rng):
        steps_per_epoch = x.shape[0] // batch_size
        batches = jax.random.permutation(rng, x.shape[0])
        batches = batches[: steps_per_epoch * batch_size]
        batches = batches.reshape((steps_per_epoch, batch_size))

        features = [self.schema["user"]["id"], self.schema["item"]["id"]]

        epoch_loss = []
        for batch in batches:
            user_id, item_id = jnp.split(x[batch][features].values, 2, axis=1)
            grads, loss = self._apply(state, user_id, item_id)
            state = self._update(state, grads)
            epoch_loss.append(loss)

        train_loss = np.mean(epoch_loss)
        return state, train_loss

    @jax.jit
    def _apply(self, state, *features):
        def loss_fn(params):
            embeddings = self.apply({"params": params}, *features)
            loss = self._loss(*embeddings)
            return loss, None

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, _), grads = grad_fn(state.params)
        return grads, loss

    @jax.jit
    @staticmethod
    def _update(state, grads):
        return state.apply_gradients(grads=grads)

    @staticmethod
    def _loss(query_embeddings, candidate_embeddings):
        def categorical_crossentropy(y_true, y_pred):
            log_probs = jax.nn.log_softmax(y_pred, axis=-1)
            loss = -jnp.sum(y_true * log_probs)
            return jnp.mean(loss)

        query_embeddings = jnp.squeeze(query_embeddings)
        candidate_embeddings = jnp.squeeze(candidate_embeddings)
        scores = jnp.matmul(query_embeddings, jnp.transpose(candidate_embeddings))

        num_queries = scores.shape[0]
        num_candidates = scores.shape[1]

        labels = jnp.eye(num_queries, num_candidates)
        loss = categorical_crossentropy(labels, scores)
        return loss

    def _train_state(self, lr, rng):
        user_id = jnp.zeros((5, 1), jnp.int32)
        item_id = jnp.zeros((5, 1), jnp.int32)
        params = self.init(rng, user_id, item_id)["params"]

        tx = optax.adam(lr)
        state = TrainState.create(apply_fn=self.apply, params=params, tx=tx)
        return state
