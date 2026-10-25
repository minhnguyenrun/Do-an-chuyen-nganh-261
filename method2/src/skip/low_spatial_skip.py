from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, init=False)
class LowSpatialSkip:
    lines: tuple
    image: np.ndarray

    def __init__(self,lines,image):
        object.__setattr__(self,'lines',tuple(lines))
        # Immutable bytes backing also prevents callers re-enabling write access.
        snapshot = np.frombuffer(image.tobytes(),dtype=image.dtype).reshape(image.shape)
        object.__setattr__(self,'image',snapshot)
