from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from .base import BaseEncoder
from ..one_hot import OneHotEncoder


class CategoryEncoder(BaseEncoder):
    def __init__(self, feature_index: list[int], name: str = None):
        pipeline = make_pipeline(OrdinalEncoder(), OneHotEncoder())
        return super().__init__(pipeline, feature_index, name)
