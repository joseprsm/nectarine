class BaseEncoder:
    def __init__(self, transformer, feature_index: list[int], name: str = None):
        self._name = name or self.__class__.__name__
        self._transformer = transformer
        self._feature_index = feature_index

    def __iter__(self):
        for x in [self._name, self._transformer, self._feature_index]:
            yield x
