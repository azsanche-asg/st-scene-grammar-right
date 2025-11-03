from importlib import import_module
from typing import Any, Dict

import numpy as np

from .base import Policy
from .heuristic import HeuristicPolicy

try:
    import torch  # optional for feature backbones
except ImportError:  # pragma: no cover - allow environments without torch
    torch = None


class LearnedPolicy(Policy):
    def __init__(self, scorer: str = "features", backbone_name: str = "dino", train: bool = False, **kwargs):
        self.heuristic = HeuristicPolicy()
        self.scorer = scorer
        self.train = train
        self.backbone_name = backbone_name
        self.backbone_fallback_error = None
        self.backbone = self._load_backbone(backbone_name)

    def _load_backbone(self, backbone_name: str):
        try:
            module = import_module(f"backbones.{backbone_name}")
            cls = getattr(module, f"{backbone_name.capitalize()}Backbone")
            return cls()
        except Exception as exc:  # pragma: no cover - fallback for missing deps
            if backbone_name != "identity":
                try:
                    module = import_module("backbones.identity")
                    cls = getattr(module, "IdentityBackbone")
                    self.backbone_name = "identity"
                    self.backbone_fallback_error = exc
                    return cls()
                except Exception:
                    raise exc
            raise

    def _frame_to_image(self, frame: Dict[str, Any], seq: Dict[str, Any]) -> np.ndarray:
        image = frame.get("image")
        if image is not None:
            return image
        height = seq.get("height", 160)
        width = seq.get("width", 256)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for rect in frame.get("rects", []):
            x, y, w_rect, h_rect = rect
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(width, x1 + int(w_rect))
            y2 = min(height, y1 + int(h_rect))
            canvas[y1:y2, x1:x2] = 255
        return canvas

    def score_sequence(self, seq: Dict[str, Any]) -> float:
        feats = []
        for frame in seq["frames"]:
            image = self._frame_to_image(frame, seq)
            emb = self.backbone.embed(image)
            emb = np.asarray(emb)
            if emb.ndim >= 2:
                emb_flat = emb.reshape(-1, emb.shape[-1]) if emb.ndim > 2 else emb
                pooled = emb_flat.mean(0)
            else:
                pooled = emb.reshape(-1).mean(keepdims=True)
            feats.append(np.atleast_1d(pooled))
        if len(feats) < 2:
            return 0.0
        stacked = np.stack(feats)
        diffs = np.linalg.norm(np.diff(stacked, axis=0), axis=1)
        return float(-np.mean(diffs))

    def induce(self, seq: Dict[str, Any]) -> Dict[str, Any]:
        grammar = self.heuristic.induce(seq)
        score = self.score_sequence(seq)
        meta = grammar.setdefault("meta", {})
        meta["score"] = score
        meta["policy"] = "learned"
        meta["backbone"] = self.backbone_name
        if self.backbone_fallback_error is not None:
            meta["backbone_fallback"] = str(self.backbone_fallback_error)
        return grammar
