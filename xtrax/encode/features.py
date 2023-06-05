from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline

from .min_max import MinMaxScaler


class BaseEncoder(tuple):
    def __new__(cls, transformer, feature_index: list[int], name: str = None):
        name = name or cls.__name__
        return tuple.__new__(cls, (name, transformer, feature_index))


class CategoryEncoder(BaseEncoder):
    def __new__(cls, feature_index: list[int], name: str = None):
        pipeline = make_pipeline(OrdinalEncoder(), OneHotEncoder())
        return super().__new__(pipeline, feature_index, name)


class NumberEncoder(BaseEncoder):
    def __new__(cls, feature_index: list[int], name: str = None):
        transformer = MinMaxScaler(feature_range=(-1, 1))
        return super().__new__(transformer, feature_index, name)


class IDEncoder(BaseEncoder):
    def __new__(cls, feature_index: list[int], name: str = None):
        transformer = OrdinalEncoder()
        return super().__new__(transformer, feature_index, name)
