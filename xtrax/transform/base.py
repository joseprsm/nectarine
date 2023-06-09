from abc import abstractmethod

from sklearn.compose import ColumnTransformer


class BaseTransformer(ColumnTransformer):
    def __init__(self, schema: dict, header: list[str], remainder: str = "drop"):
        self._schema = schema
        self._header = header
        transformers = self._get_transformers(self._schema, self._header)
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
