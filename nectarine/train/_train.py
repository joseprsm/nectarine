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


rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)


class Train:
    params: dict
    model: TwoTower
    state: TrainState

    def __init__(self, config: dict) -> None:
        self._config: dict[str, dict] = config["train"]

    def __enter__(self):
        self.params = ...  # todo: validate parameters
        self.model = self._set_model(self._config)
        self.state = self._create_train_state()
        return self

    def __call__(
        self,
        training_data,
    ):
        for epoch in range(1, self.params["epochs"] + 1):
            state, train_loss = self._train_epoch(training_data)
            print(f"epoch: {epoch}, train_loss: {train_loss}")
        return state

    def __exit__(self, *_):
        pass

    def _train_epoch(self, data: pd.DataFrame):
        steps_per_epoch = data.shape[0] // self.params["batch_size"]
        batches = jax.random.permutation(rng, data.shape[0])
        batches = batches[: steps_per_epoch * self.params["batch_size"]]
        batches = batches.reshape((steps_per_epoch, self.params["batch_size"]))

        epoch_loss = []

        for batch in batches:
            batch = json.loads(data.iloc[batch, :].to_json(orient="records"))
            grads, loss = self.model.retrieval(self.state, **batch)
            state = self.state.apply_gradients(grads=grads)
            epoch_loss.append(loss)

        train_loss = jnp.mean(jnp.array(epoch_loss))
        return state, train_loss

    def create_train_state(rng, model: nn.Module, learning_rate: float = 0.01):
        inputs = ...  # todo: get sample inputs
        params = model.init(rng, **inputs)["params"]
        tx = optax.adamw(learning_rate)
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @staticmethod
    def _set_model(config: dict):
        def is_module(x):
            return "flax.linen.module.Module" in str(x[1].__base__)

        model_config: dict = ...  # todo: get model config
        module = importlib.import_module(f"nectarine.train.models.{config['model']}")
        model = filter(is_module, inspect.getmembers(module, inspect.isclass))
        model = next(model)[1](model_config)
        return model
