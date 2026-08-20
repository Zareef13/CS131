"""Hybrid obstacle-awareness risk fusion."""

from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from . import config
from .motion import compute_motion_score
from .preprocessing import normalize_map, preprocess_frame
from .structure import compute_structure_scores
from .profiling import timed_stage


class RiskResult(TypedDict):
    overlay: NDArray[np.uint8]
    risk_display: NDArray[np.float32]
    motion_score: NDArray[np.float32]
    edge_map: NDArray[np.float32]
    line_map: NDArray[np.float32]
    canny_edges: NDArray[np.uint8]


def compute_hybrid_risk(
    frame1: NDArray[np.uint8], frame2: NDArray[np.uint8]
) -> RiskResult:
    """Fuse motion, edge, and line cues into a heuristic visual risk map."""
    gray1, canny_edges, edge_map, line_map = compute_structure_scores(frame1)
    gray2, _ = preprocess_frame(frame2)
    motion_score = compute_motion_score(gray1, gray2)

    with timed_stage("risk_map_fusion"):
        risk = (
            config.MOTION_WEIGHT * motion_score
            + config.EDGE_WEIGHT * edge_map
            + config.LINE_WEIGHT * line_map
        )
    with timed_stage("thresholding_postprocessing"):
        risk[risk < np.percentile(risk, config.RISK_PERCENTILE)] = 0
        risk_display = normalize_map(cv2.GaussianBlur(risk, (41, 41), 0))

    with timed_stage("overlay_rendering"):
        heatmap = cv2.applyColorMap(
            (risk_display * 255).astype(np.uint8), cv2.COLORMAP_HOT
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(frame1, 0.50, heatmap, 0.40, 0)

    return {
        "overlay": overlay,
        "risk_display": risk_display,
        "motion_score": motion_score,
        "edge_map": edge_map,
        "line_map": line_map,
        "canny_edges": canny_edges,
    }
