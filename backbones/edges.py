import cv2
import numpy as np

from .base import Backbone


class EdgesBackbone(Backbone):
    def embed(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges[..., None].astype(np.float32) / 255.0
