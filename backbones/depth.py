import numpy as np
import torch
import cv2
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

from .base import Backbone


class DepthBackbone(Backbone):
    def __init__(self, model_name: str = "Intel/dpt-hybrid-midas"):
        self.extractor = DPTFeatureExtractor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.eval()

    def embed(self, image: np.ndarray) -> np.ndarray:
        inputs = self.extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
        return depth[..., None]
