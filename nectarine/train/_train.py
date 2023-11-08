import importlib
import inspect
import json

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from flax import linen as nn
from flax.training.train_state import TrainState

from nectarine.train.models import TwoTower


NUM_EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32

rng = jax.random.PRNGKey(0)
rng, INIT_RNG = jax.random.split(rng)


class Train:
    params: dict
    model: TwoTower
    state: TrainState

    def __init__(self, config: dict) -> None:
        self.config = config["train"].copy()
        self.params = config.pop("hyperparameters")

    def __enter__(self):
        return self

    def __call__(
        self,
        training_data,
    ):
        def get_param(value: str):
            return (
                self.params[value]
                if value in self.params.keys()
                else globals()[value.upper()]
            )

        epochs = get_param("num_epochs")
        lr = get_param("learning_rate")
        batch_size = get_param("batch_size")

        self.model = self._set_model(self.config["model"], self.params)

        self.state = self._init(self.model, lr)
        for epoch in range(1, epochs + 1):
            self.state, train_loss = self._train_epoch(training_data, batch_size)
            print(f"epoch: {epoch}, train_loss: {train_loss}")

        return self.state

    def __exit__(self, *_):
        pass

    def _train_epoch(self, data: pd.DataFrame, batch_size: int):
        steps_per_epoch = data.shape[0] // batch_size
        batches = jax.random.permutation(rng, data.shape[0])
        batches = batches[: steps_per_epoch * batch_size]
        batches = batches.reshape((steps_per_epoch, batch_size))

        epoch_loss = []

        for batch in batches:
            batch = json.loads(data.iloc[batch, :].to_json(orient="records"))
            grads, loss = self.model.retrieval(self.state, **batch)
            state = self.state.apply_gradients(grads=grads)
            epoch_loss.append(loss)

        train_loss = jnp.mean(jnp.array(epoch_loss))
        return state, train_loss

    @jax.jit
    def _init(self, learning_rate: float = 0.01):
        inputs = {
            "query": {"id": jnp.zeros((5, 1))},
            "candidate": {"id": jnp.zeros((5, 1))},
        }
        params = self.model.init(INIT_RNG, **inputs)["params"]
        tx = optax.adamw(learning_rate)
        return TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    @staticmethod
    def _set_model(model_name: str, params: dict) -> nn.Module:
        def is_module(x):
            return "flax.linen.module.Module" in str(x[1].__base__)

        module = f"nectarine.train.models.{model_name}"
        module = importlib.import_module(module)

        model = filter(is_module, inspect.getmembers(module, inspect.isclass))
        model = next(model)[1](**params)
        return model
