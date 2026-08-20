"""Frame preprocessing and numerical helpers."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .profiling import timed_stage


def normalize_map(values: NDArray[np.generic]) -> NDArray[np.float32]:
    """Min-max normalize an array to approximately [0, 1]."""
    values = values.astype(np.float32)
    minimum = np.min(values)
    maximum = np.max(values)
    return (values - minimum) / (maximum - minimum + 1e-8)


def preprocess_frame(
    frame: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Convert an RGB frame to grayscale and apply the original blur."""
    with timed_stage("grayscale_conversion"):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    with timed_stage("gaussian_blur"):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred
