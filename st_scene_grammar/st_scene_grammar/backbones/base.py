from abc import ABC, abstractmethod
import numpy as np
class Backbone(ABC):
    @abstractmethod
    def embed(self, image: np.ndarray) -> np.ndarray:
        pass
