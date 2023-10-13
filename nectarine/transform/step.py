from ..step import Step
from .transformer import FeatureTransformer


class Transform(Step):
    def __init__(self, schema: dict, target: str):
        super().__init__()
        self.schema = schema
        self.target = target
        self.transformer = FeatureTransformer(self.schema[self.target])

    def __call__(self, x):
        return self.transformer.fit(x).transform(x)

    def download(self):
        pass
