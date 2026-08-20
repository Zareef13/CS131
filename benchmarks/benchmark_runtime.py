"""Benchmark pipeline-only and complete decode/process/encode runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2

from risksight.pipeline import compute_hybrid_risk
from risksight.profiling import collect_stage_timings
from risksight.video import load_video_frames

from .common import machine_metadata, resize_rgb, timing_summary, video_info, warm_up, write_csv, write_json, OUTPUT_DIR


def pipeline_benchmark(path: Path, width: int, max_frames: int | None, warmup_frames: int):
    frames = load_video_frames(path, max_frames=max_frames, resize_width=width)
    if len(frames) < 2:
        raise ValueError("benchmark requires at least two decoded frames")
    warm_up(frames, compute_hybrid_risk, warmup_frames)
    durations = []
    with collect_stage_timings() as stages:
        for first, second in zip(frames, frames[1:]):
            start = perf_counter()
            compute_hybrid_risk(first, second)
            durations.append(perf_counter() - start)
    total_pipeline = sum(durations)
    stage_rows = []
    for name, values in sorted(stages.items()):
        summary = timing_summary(values)
        stage_rows.append({
            "stage": name,
            **summary,
            "runtime_percent": 100 * sum(values) / total_pipeline,
        })
    return timing_summary(durations), stage_rows


def end_to_end_benchmark(path: Path, width: int, max_frames: int | None):
    capture = cv2.VideoCapture(str(path))
    success, first_bgr = capture.read()
    if not success:
        capture.release()
        raise ValueError(f"Could not decode video: {path}")
    first = resize_rgb(first_bgr, width)
    height = first.shape[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{path.stem.lower()}_benchmark_overlay.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not open benchmark writer: {output_path}")
    durations = []
    try:
        while max_frames is None or len(durations) < max_frames - 1:
            start = perf_counter()
            success, second_bgr = capture.read()
            if not success:
                break
            second = resize_rgb(second_bgr, width)
            result = compute_hybrid_risk(first, second)
            writer.write(cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
            durations.append(perf_counter() - start)
            first = second
    finally:
        writer.release()
        capture.release()
    if not durations:
        raise ValueError("benchmark requires at least two decoded frames")
    return timing_summary(durations), str(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path, default=[Path("data/Drone.mp4"), Path("data/car.MP4")])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=5)
    args = parser.parse_args(argv)
    videos = []
    for path in args.videos:
        print(f"Benchmarking {path} ...", flush=True)
        info = video_info(path, args.width)
        pipeline, stages = pipeline_benchmark(path, args.width, args.max_frames, args.warmup_frames)
        end_to_end, artifact = end_to_end_benchmark(path, args.width, args.max_frames)
        videos.append({**info, "warmup_frames": args.warmup_frames, "pipeline_only": pipeline, "end_to_end": end_to_end, "stages": stages, "overlay_artifact": artifact})
    payload = {"environment": machine_metadata(), "videos": videos}
    write_json("runtime.json", payload)
    write_csv("runtime.csv", [{"video": x["video"], "frames": x["pipeline_only"]["samples"], "source_resolution": x["source_resolution"], "processing_resolution": x["processing_resolution"], "pipeline_fps": x["pipeline_only"]["fps"], "end_to_end_fps": x["end_to_end"]["fps"], "pipeline_mean_ms": x["pipeline_only"]["mean_ms"], "pipeline_median_ms": x["pipeline_only"]["median_ms"], "pipeline_p95_ms": x["pipeline_only"]["p95_ms"], "pipeline_total_seconds": x["pipeline_only"]["total_seconds"], "end_to_end_total_seconds": x["end_to_end"]["total_seconds"]} for x in videos])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
