from flax import linen as nn

from .lookup import LookupModule


_MAPPING = {"id": LookupModule, "category": LookupModule}


class TransformLayer(nn.Module):
    schema: dict
    config: dict

    def setup(self):
        self._layers = {
            feature: _MAPPING[feature_type](self.config[feature])
            for feature, feature_type in self.schema.items()
        }

    def __call__(self, x, *_):
        return {feature: layer(x[feature]) for feature, layer in self._layers.items()}
