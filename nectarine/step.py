from abc import ABC


class Step(ABC):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __call__(self):
        raise NotImplementedError

    def download(self, **kwargs):
        raise NotImplementedError
