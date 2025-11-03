import glob
import os
from typing import Any, Dict, Iterator

import cv2


class RE10KDataset:
    """
    Adapter for the RealEstate10K dataset (RGB frames).
    Expects a folder structure like:
        root/
          video_001/frame_0000.jpg
          video_001/frame_0001.jpg
          ...
    """

    def __init__(self, root: str, seq_len: int = 12, stride: int = 2):
        self.root = root
        self.seq_len = seq_len
        self.stride = stride
        self.video_dirs = sorted(
            [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for vid in self.video_dirs:
            frames = sorted(glob.glob(os.path.join(vid, "*.jpg")))
            if len(frames) < self.seq_len:
                continue
            for i in range(0, len(frames) - self.seq_len + 1, self.stride):
                seq_paths = frames[i : i + self.seq_len]
                seq_frames = []
                for p in seq_paths:
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    seq_frames.append({
                        "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        "rects": [],
                    })
                if len(seq_frames) != self.seq_len:
                    continue
                height, width = seq_frames[0]["image"].shape[:2]
                seq = {
                    "frames": seq_frames,
                    "width": width,
                    "height": height,
                    "gt": {"repeat_axis": None, "regular": False},
                }
                yield seq


class ScanNetPPDataset:
    """
    Adapter for ScanNet++ or ScanNet-like RGB frame folders.
    Similar to RE10K but can read .png and .jpeg files.
    """

    def __init__(self, root: str, seq_len: int = 12, stride: int = 2):
        self.root = root
        self.seq_len = seq_len
        self.stride = stride
        self.scene_dirs = sorted(
            [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for scene in self.scene_dirs:
            frames = sorted(glob.glob(os.path.join(scene, "*.png")))
            frames += sorted(glob.glob(os.path.join(scene, "*.jpg")))
            frames += sorted(glob.glob(os.path.join(scene, "*.jpeg")))
            if len(frames) < self.seq_len:
                continue
            for i in range(0, len(frames) - self.seq_len + 1, self.stride):
                seq_paths = frames[i : i + self.seq_len]
                seq_frames = []
                for p in seq_paths:
                    img = cv2.imread(p)
                    if img is None:
                        continue
                    seq_frames.append({
                        "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        "rects": [],
                    })
                if len(seq_frames) != self.seq_len:
                    continue
                height, width = seq_frames[0]["image"].shape[:2]
                seq = {
                    "frames": seq_frames,
                    "width": width,
                    "height": height,
                    "gt": {"repeat_axis": None, "regular": False},
                }
                yield seq
