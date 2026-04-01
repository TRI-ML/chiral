from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ProprioConfig:
    """Static description of one proprioception stream.

    Passed to ``PolicyServer.__init__`` so the base class can pre-allocate
    ``self.proprios`` before any step is taken.
    """
    name:  str
    size:  int              # number of elements
    dtype: np.dtype = np.float32


@dataclass
class CameraConfig:
    """Static description of one camera channel.

    Passed to ``PolicyServer.__init__`` so the base class can pre-allocate
    ``self.images`` and ``self.depths`` before any step is taken.
    """
    name:        str
    height:      int
    width:       int
    channels:    int        = 3
    has_depth:   bool       = False
    intrinsics:  np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    extrinsics:  np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    image_dtype: np.dtype   = np.uint8
    depth_dtype: np.dtype   = np.float32


@dataclass
class CameraInfo:
    name: str
    intrinsics: np.ndarray          # (3, 3) float64 — [fx,0,cx; 0,fy,cy; 0,0,1]
    extrinsics: np.ndarray          # (4, 4) float64 — camera-to-world
    image: np.ndarray               # (H, W, C) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32, metres
    timestamp: float = 0.0          # monotonic time when this frame was captured


@dataclass
class Observation:
    cameras:  list[CameraInfo]
    proprios: dict[str, np.ndarray] = field(default_factory=dict)  # name → 1-D array
    timestamp: float = 0.0
    extra: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> CameraInfo:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(name)
