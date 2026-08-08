"""Canny edge and Hough line feature extraction."""

import cv2
import numpy as np
from numpy.typing import NDArray

from . import config
from .preprocessing import normalize_map, preprocess_frame


def detect_edges(blurred: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply Canny using the original thresholds."""
    return cv2.Canny(
        blurred, config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD
    )


def detect_lines(edges: NDArray[np.uint8]):
    """Apply the probabilistic Hough transform with original parameters."""
    return cv2.HoughLinesP(
        edges,
        rho=config.HOUGH_RHO,
        theta=np.pi / config.HOUGH_THETA_DIVISOR,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP,
    )


def compute_structure_scores(
    frame: NDArray[np.uint8],
) -> tuple[
    NDArray[np.uint8],
    NDArray[np.uint8],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """Compute grayscale, binary edge, edge-influence, and line maps."""
    gray, blurred = preprocess_frame(frame)
    canny_edges = detect_edges(blurred)

    edge_map = normalize_map(canny_edges)
    edge_map = cv2.dilate(edge_map, np.ones((5, 5), np.uint8), iterations=1)
    edge_map = normalize_map(cv2.GaussianBlur(edge_map, (11, 11), 0))

    line_map = np.zeros_like(edge_map)
    detected_lines = detect_lines(canny_edges)
    if detected_lines is not None:
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_map, (x1, y1), (x2, y2), 1.0, 2)

    line_map = normalize_map(cv2.GaussianBlur(line_map, (9, 9), 0))
    return gray, canny_edges, edge_map, line_map
