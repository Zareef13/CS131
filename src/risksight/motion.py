"""Dense optical-flow motion estimation."""

import cv2
import numpy as np
from numpy.typing import NDArray

from . import config
from .preprocessing import normalize_map


def compute_motion_score(
    gray1: NDArray[np.uint8], gray2: NDArray[np.uint8]
) -> NDArray[np.float32]:
    """Compute a normalized dense Farneback flow-magnitude map."""
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=config.FLOW_PYR_SCALE,
        levels=config.FLOW_LEVELS,
        winsize=config.FLOW_WINDOW_SIZE,
        iterations=config.FLOW_ITERATIONS,
        poly_n=config.FLOW_POLY_N,
        poly_sigma=config.FLOW_POLY_SIGMA,
        flags=0,
    )
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    motion_score = normalize_map(magnitude)
    return normalize_map(cv2.GaussianBlur(motion_score, (11, 11), 0))
