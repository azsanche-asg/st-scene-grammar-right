import numpy as np
from .base import Backbone
class IdentityBackbone(Backbone):
    def embed(self, image: np.ndarray) -> np.ndarray:
        gray = image.mean(axis=2, keepdims=True).astype(np.float32) / 255.0
        return gray
