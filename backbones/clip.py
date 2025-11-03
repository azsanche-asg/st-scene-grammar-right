import numpy as np
import torch
import open_clip

from .base import Backbone


class ClipBackbone(Backbone):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()

    def embed(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(image.astype("uint8"))
        x = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feat = self.model.encode_image(x)
        return feat.cpu().numpy()
