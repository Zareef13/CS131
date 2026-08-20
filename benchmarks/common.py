"""Shared, deterministic helpers for RiskSight benchmarks."""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "benchmarks"


def percentile(values: Iterable[float], q: float) -> float:
    """Return a linearly interpolated percentile, including singleton input."""
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        raise ValueError("at least one timing is required")
    return float(np.percentile(data, q))


def timing_summary(seconds: Iterable[float]) -> dict[str, float | int]:
    values = list(seconds)
    if not values:
        raise ValueError("at least one timing is required")
    total = float(sum(values))
    return {
        "samples": len(values),
        "total_seconds": total,
        "mean_ms": statistics.fmean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": percentile(values, 95) * 1000,
        "fps": len(values) / total if total else float("inf"),
    }


def machine_metadata() -> dict[str, str | bool]:
    cpu = platform.processor() or "unknown"
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                detected = subprocess.run(
                    ["sysctl", "-n", key], check=True, capture_output=True, text=True
                ).stdout.strip()
                if detected:
                    cpu = detected
                    break
            except (OSError, subprocess.SubprocessError):
                continue
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "gpu_acceleration": False,
        "timer": "time.perf_counter",
    }


def resize_rgb(frame_bgr: np.ndarray, width: int) -> np.ndarray:
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def video_info(path: Path, width: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return {
        "video": str(path),
        "source_resolution": f"{source_width}x{source_height}",
        "processing_resolution": f"{width}x{int(source_height * width / source_width)}",
        "source_fps": fps,
    }


def warm_up(frames: list[np.ndarray], function: Callable[[np.ndarray, np.ndarray], Any], count: int) -> None:
    for index in range(min(count, len(frames) - 1)):
        function(frames[index], frames[index + 1])


def write_json(filename: str, payload: Any) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return path


def write_csv(filename: str, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def measured_call(function: Callable[[], Any]) -> tuple[Any, float]:
    start = perf_counter()
    value = function()
    return value, perf_counter() - start
