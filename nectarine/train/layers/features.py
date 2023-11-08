from flax import linen as nn


class FeaturesModel(nn.Module):
    layer_sizes: list[int]
    activation: str = None

    @nn.compact
    def __call__(self, features):
        x = features
        for layer in self.layer_sizes:
            x = nn.Dense(layer)(x)
            if self.activation:
                x = getattr(nn, self.activation)(x)
        return x
