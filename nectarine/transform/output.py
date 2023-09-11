import jax
import jax.numpy as jnp
from flax import linen as nn


class Transform(nn.Module):
    users: jnp.ndarray
    items: jnp.ndarray

    @nn.compact
    def __call__(
        self, user_id: jnp.ndarray, item_id: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        user_features = self._get_features(user_id, self.users)
        item_features = self._get_features(item_id, self.items)
        # todo: include context features
        return user_features, item_features

    @staticmethod
    @jax.jit
    def _get_features(ids: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        ids = ids.reshape(-1, 1)
        features = jnp.take(data, ids, axis=0)
        # todo: confirm features shape is always (1, 1, ..., n)
        if features.shape[-1] != 0:
            features = features.reshape(-1, features.shape[-1])
            return jnp.concatenate([ids, features], axis=1)
        return ids

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"users={self.users.shape}, "
            f"items={self.items.shape})"
        )
