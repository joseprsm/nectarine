from abc import abstractmethod

import numpy as np
from sklearn.compose import ColumnTransformer


class BaseTransformer(ColumnTransformer):
    def __init__(self, schema: dict, header: list[str] = None, remainder: str = "drop"):
        self._schema = schema
        self._header = header
        transformers = self._get_transformers(self._schema, self._header)
        transformers = self._first_transformer_id(transformers)
        super().__init__(transformers=transformers, remainder=remainder)

    @abstractmethod
    def _get_transformers(cls, schema: dict, header: list[str]):
        raise NotImplementedError

    @property
    def header(self):
        return self._header

    @property
    def schema(self):
        return self._schema

    @staticmethod
    def _first_transformer_id(transformers: list):
        id_encoder = transformers.pop(
            np.where([x[0] == "IDEncoder" for x in transformers])[0][0]
        )
        return [id_encoder] + transformers
