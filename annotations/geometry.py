"""Coordinate conversion and mask validation helpers."""

from __future__ import annotations

import numpy as np


def normalized_box_to_pixels(box: list[float], width: int, height: int) -> list[int]:
    if len(box) != 4 or width < 1 or height < 1:
        raise ValueError("box must contain four values and image dimensions must be positive")
    if any(not 0 <= value <= 1 for value in box):
        raise ValueError("normalized box values must be in [0, 1]")
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box must have positive area")
    return [round(x0 * (width - 1)), round(y0 * (height - 1)), round(x1 * (width - 1)), round(y1 * (height - 1))]


def normalized_point_to_pixels(point: list[float], width: int, height: int) -> list[int]:
    if len(point) != 2 or any(not 0 <= value <= 1 for value in point):
        raise ValueError("normalized point must contain two values in [0, 1]")
    return [round(point[0] * (width - 1)), round(point[1] * (height - 1))]


def merge_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        if mask.shape != shape:
            raise ValueError("all masks must match the requested shape")
        merged[mask.astype(bool)] = 255
    return merged


def rejection_reasons(mask: np.ndarray, box_pixels: list[int], confidence: float, minimum_confidence: float = 0.5) -> list[str]:
    reasons: list[str] = []
    binary = mask.astype(bool)
    area = int(binary.sum())
    height, width = binary.shape
    x0, y0, x1, y1 = box_pixels
    box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    if confidence < minimum_confidence:
        reasons.append("gemini_confidence_below_threshold")
    if area == 0:
        reasons.append("empty_mask")
        return reasons
    if area / binary.size > 0.70:
        reasons.append("mask_over_70_percent")
    inside = int(binary[y0:y1 + 1, x0:x1 + 1].sum())
    if inside / area < 0.50:
        reasons.append("mostly_outside_gemini_box")
    if area / box_area < 0.01:
        reasons.append("mask_tiny_relative_to_box")
    border = np.zeros_like(binary)
    border_height = max(1, round(height * 0.05)); border_width = max(1, round(width * 0.05))
    border[:border_height] = True; border[-border_height:] = True
    border[:, :border_width] = True; border[:, -border_width:] = True
    if int((binary & border).sum()) / area > 0.85 and area / binary.size < 0.05:
        reasons.append("possible_hud_or_border_artifact")
    return reasons
