"""Demo image and overlay-video generation."""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np

from . import config
from .motion import compute_motion_score
from .pipeline import compute_hybrid_risk
from .preprocessing import preprocess_frame
from .structure import detect_edges, detect_lines


def _save_figure(figure, output_path: Path) -> None:
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_sample_frames(
    frames: list[NDArray[np.uint8]], output_path: Path
) -> None:
    figure, axes = plt.subplots(1, len(config.SAMPLE_FRAME_INDICES), figsize=(20, 5))
    for axis, requested_index in zip(axes, config.SAMPLE_FRAME_INDICES):
        index = min(requested_index, len(frames) - 1)
        axis.imshow(frames[index])
        axis.set_title(f"Frame {index}")
        axis.axis("off")
    _save_figure(figure, output_path)


def save_edge_demo(frames: list[NDArray[np.uint8]], output_path: Path) -> None:
    frame = frames[min(config.DEMO_FRAME_INDEX, len(frames) - 1)]
    _, blurred = preprocess_frame(frame)
    edges = detect_edges(blurred)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, image, title, cmap in (
        (axes[0], frame, "Original", None),
        (axes[1], blurred, "Grayscale + Blur", "gray"),
        (axes[2], edges, "Canny Edges", "gray"),
    ):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    _save_figure(figure, output_path)


def save_hough_demo(frames: list[NDArray[np.uint8]], output_path: Path) -> None:
    frame = frames[min(config.DEMO_FRAME_INDEX, len(frames) - 1)]
    _, blurred = preprocess_frame(frame)
    edges = detect_edges(blurred)
    line_image = frame.copy()
    lines = detect_lines(edges)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, image, title, cmap in (
        (axes[0], frame, "Original", None),
        (axes[1], edges, "Canny Edges", "gray"),
        (axes[2], line_image, "Hough Lines", None),
    ):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    _save_figure(figure, output_path)


def save_optical_flow_demo(
    frames: list[NDArray[np.uint8]], output_path: Path
) -> None:
    index = min(config.DEMO_FRAME_INDEX, len(frames) - 2)
    gray1, _ = preprocess_frame(frames[index])
    gray2, _ = preprocess_frame(frames[index + 1])
    motion_score = compute_motion_score(gray1, gray2)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, image, title, cmap in (
        (axes[0], frames[index], f"Frame {index}", None),
        (axes[1], frames[index + 1], f"Frame {index + 1}", None),
        (axes[2], motion_score, "Optical Flow Magnitude", "hot"),
    ):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    _save_figure(figure, output_path)


def save_risk_overlay_demo(
    frames: list[NDArray[np.uint8]], output_path: Path
) -> None:
    index = min(config.DEMO_FRAME_INDEX, len(frames) - 2)
    result = compute_hybrid_risk(frames[index], frames[index + 1])
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for axis, image, title, cmap in (
        (axes[0], frames[index], "Original", None),
        (axes[1], result["motion_score"], "Motion Score", "hot"),
        (axes[2], result["risk_display"], "Hybrid Risk Map", "hot"),
        (axes[3], result["overlay"], "Hybrid Risk Overlay", None),
    ):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    _save_figure(figure, output_path)


def save_risk_overlay_video(
    frames: list[NDArray[np.uint8]], output_path: Path
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.OUTPUT_FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open output video writer: {output_path}")
    try:
        for index in range(len(frames) - 1):
            result = compute_hybrid_risk(frames[index], frames[index + 1])
            writer.write(cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
            if index % 50 == 0:
                print(f"Processed frame {index}/{len(frames) - 1}")
    finally:
        writer.release()


def save_all_outputs(
    frames: list[NDArray[np.uint8]], output_dir: Path, prefix: str
) -> None:
    """Generate the same six artifacts as the original implementation."""
    if len(frames) < 2:
        raise ValueError("At least two frames are required to generate outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = (
        (save_sample_frames, f"{prefix}_sample_frames.png"),
        (save_edge_demo, f"{prefix}_edge_demo.png"),
        (save_hough_demo, f"{prefix}_hough_demo.png"),
        (save_optical_flow_demo, f"{prefix}_optical_flow_demo.png"),
        (save_risk_overlay_demo, f"{prefix}_risk_overlay_frame_40.png"),
        (save_risk_overlay_video, f"{prefix}_risk_overlay_video.mp4"),
    )
    for function, filename in tasks:
        function(frames, output_dir / filename)
        print(f"Saved {output_dir / filename}")
