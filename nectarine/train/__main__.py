import json
import pickle

import click
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.training.train_state import TrainState

from nectarine.train import Recommender


NUM_EPOCHS = 10
BATCH_SIZE = 500
LEARNING_RATE = 0.01
EMBEDDING_DIM = 8
TEST_SIZE = 0.3

RNG = jax.random.PRNGKey(0)


def create_train_state(model, rng):
    user_id = jnp.zeros((5, 1), jnp.int32)
    item_id = jnp.zeros((5, 1), jnp.int32)
    params = model.init(rng, user_id, item_id)["params"]

    tx = optax.adam(LEARNING_RATE)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


@jax.jit
def apply_model(model, state, user_id, item_id):
    def categorical_crossentropy(y_true, y_pred):
        log_probs = jax.nn.log_softmax(y_pred, axis=-1)
        loss = -jnp.sum(y_true * log_probs)
        return jnp.mean(loss)

    def loss_fn(params):
        query_embeddings, candidate_embeddings = model.apply(
            {"params": params}, user_id, item_id
        )
        query_embeddings = jnp.squeeze(query_embeddings)
        candidate_embeddings = jnp.squeeze(candidate_embeddings)
        scores = jnp.matmul(query_embeddings, jnp.transpose(candidate_embeddings))

        num_queries = scores.shape[0]
        num_candidates = scores.shape[1]

        labels = jnp.eye(num_queries, num_candidates)
        loss = categorical_crossentropy(labels, scores)
        return loss, jnp.array([0.0])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(model, state, train_data: pd.DataFrame, rng):
    steps_per_epoch = train_data.shape[0] // BATCH_SIZE
    batches = jax.random.permutation(rng, train_data.shape[0])
    batches = batches[: steps_per_epoch * BATCH_SIZE]
    batches = batches.reshape((steps_per_epoch, BATCH_SIZE))

    epoch_loss = []

    for batch in batches:
        user_id, item_id = jnp.split(train_data.iloc[batch][[0, 1]].values, 2, axis=1)
        grads, loss = apply_model(model, state, user_id, item_id)
        state = update_model(state, grads)
        epoch_loss.append(loss)

    train_loss = np.mean(epoch_loss)
    return state, train_loss


def test_epoch(model, state, validation_data: pd.DataFrame, rng):
    steps_per_epoch = validation_data.shape[0] // BATCH_SIZE
    batches = jax.random.permutation(rng, validation_data.shape[0])
    batches = batches[: steps_per_epoch * BATCH_SIZE]
    batches = batches.reshape((steps_per_epoch, BATCH_SIZE))

    epoch_loss = []

    for batch in batches:
        user_id, item_id = jnp.split(
            validation_data.iloc[batch][[0, 1]].values, 2, axis=1
        )
        _, loss = apply_model(model, state, user_id, item_id)
        epoch_loss.append(loss)

    test_loss = np.mean(epoch_loss)
    return test_loss


def train_and_evaluate(model, state, train_data, validation_data, rng):
    for epoch in range(1, NUM_EPOCHS + 1):
        state, train_loss = train_epoch(model, state, train_data, rng)
        test_loss = test_epoch(model, state, validation_data, rng)
        print(f"epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}")
    return state


@click.command()
@click.option("--encoded-path", "--encoded", "encoded", required=True)
@click.option("--schema-path", "--schema", "schema", required=True)
@click.option("--transform-layer", "--transform", "transform_layer", required=True)
@click.option("--model-config", "--config", "model_config")
@click.option("--model-path", "model")
def train(
    encoded: str,
    schema: str,
    transform_layer: str,
    model_config: str = None,
    model_path: str = None,
):
    with open(transform_layer, "rb") as fp:
        transform_layer = pickle.load(fp)

    with open(schema, "r") as fp:
        schema = json.load(fp)

    with open(model_config, "r") as fp:
        model_config = json.load(fp)

    df = pd.read_csv(encoded).sample(frac=1, random_state=1)
    cutoff = np.floor(df.shape[0] * TEST_SIZE).astype(int)
    validation_data, train_data = df.iloc[:cutoff], df.iloc[cutoff:]

    model = Recommender(config=model_config, transform=transform_layer)

    rng, init_rng = jax.random.split(RNG)
    state = create_train_state(model, init_rng)
    state = train_and_evaluate(model, state, train_data, validation_data, rng)

    with open(model_path, "wb") as fp:
        pickle.dump(model, fp)


if __name__ == "__main__":
    train()
