from ..step import Step
from .models import Recommender


class Train(Step):
    def __init__(self, config: dict) -> None:
        self._model_config = config
        self.model = Recommender(config=self._model_config)

    def __call__(
        self, x, validation_data=None, batch_size: int = None, epochs: int = 1
    ) -> Recommender:
        self.model.compile()

        _ = self.model.fit(
            x,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=None,
        )
        return self.model

    def download(self, split: bool = True):
        raise NotImplementedError
