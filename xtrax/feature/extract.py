from xtrax.data import Dataset, Users, Items
from xtrax.feature.base import BaseTransformer
from xtrax.feature.transform import FeatureTransformer
from xtrax.transform.context import ContextPipeline


class FeatureExtractor(BaseTransformer):
    def __init__(self, schema: dict, users: str, items: str, header: list[str]):
        self._users = users
        self._items = items

        self._user_transformer = FeatureTransformer(schema["user"])
        self._item_transformer = FeatureTransformer(schema["item"])
        self._context_pipeline = ContextPipeline(schema)

        transformers = self._get_transformers(schema, header)
        super().__init__(transformers=transformers, remainder="passthrough")

    def fit(self, X):
        def fit_transformer(inputs: Dataset, transformer: FeatureTransformer):
            input_attr: str = getattr(self, f"_{inputs.__name__.lower()}")
            x = inputs.load(input_attr, schema=self._schema)
            transformer.fit(x)

        fit_transformer(Users)
        fit_transformer(Items)

        self._user_transformer.encode(X)
        self._item_transformer.encode(X)

        super().fit(X)

    def _get_transformers(self, schema: dict, header: list[str]):
        context_header = self._get_context_header(schema, header)
        return [
            (
                self._context_pipeline.__class__.__name__,
                self._context_pipeline,
                context_header,
            )
        ]

    def _get_context_header(cls, schema: dict, header: list[str]):
        raise NotImplementedError
