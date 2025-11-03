import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from .base import Backbone


class DinoBackbone(Backbone):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        feat = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        return feat
