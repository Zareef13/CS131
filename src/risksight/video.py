"""Video input and output helpers."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import RESIZE_WIDTH


def load_video_frames(
    video_path: Path,
    max_frames: int | None = None,
    resize_width: int = RESIZE_WIDTH,
) -> list[NDArray[np.uint8]]:
    """Load a video as resized RGB frames.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If OpenCV cannot open or decode the video.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: list[NDArray[np.uint8]] = []
    while max_frames is None or len(frames) < max_frames:
        success, frame = capture.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        resized_height = int(height * resize_width / width)
        frame = cv2.resize(
            frame, (resize_width, resized_height), interpolation=cv2.INTER_AREA
        )
        frames.append(frame)

    capture.release()
    if not frames:
        raise ValueError(f"No frames could be decoded from: {video_path}")
    return frames
