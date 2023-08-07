from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from ..encode.one_hot import OneHotEncoder
from .base import BaseEncoder


class CategoryEncoder(BaseEncoder):
    def __init__(self, feature_index: list[int], name: str = None):
        pipeline = make_pipeline(OrdinalEncoder(), OneHotEncoder())
        return super().__init__(pipeline, feature_index, name)
