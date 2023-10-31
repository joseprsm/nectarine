import pandas as pd

from .models import Recommender


class Train:
    def __init__(self, config: dict) -> None:
        self._config = config

    def __enter__(self):
        return self

    def __call__(
        self,
        model: Recommender,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame = None,
    ) -> Recommender:
        hyperparameters = self._config["model"]["hyperparameters"]

        _ = model.fit(
            training_data,
            validation_data=validation_data,
            callbacks=None,
            **hyperparameters,
        )
        return self.model

    def __exit__(self, *_):
        pass
