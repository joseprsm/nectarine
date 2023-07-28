from ..encode.min_max import MinMaxScaler
from .base import BaseEncoder


class NumberEncoder(BaseEncoder):
    def __init__(self, feature_index: list[int], name: str = None):
        transformer = MinMaxScaler(feature_range=(-1, 1))
        return super().__init__(transformer, feature_index, name)
