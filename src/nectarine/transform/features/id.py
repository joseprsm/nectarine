from sklearn.preprocessing import OrdinalEncoder

from .base import BaseEncoder


class IDEncoder(BaseEncoder):
    def __init__(cls, feature_index: list[int], name: str = None):
        transformer = OrdinalEncoder()
        return super().__init__(transformer, feature_index, name)
