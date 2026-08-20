"""Measure runtime of cue configurations without making accuracy claims."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from risksight.motion import compute_motion_score
from risksight.pipeline import compute_hybrid_risk
from risksight.preprocessing import preprocess_frame
from risksight.structure import detect_edges, detect_lines
from risksight.video import load_video_frames

from .common import machine_metadata, timing_summary, video_info, write_csv, write_json


def motion(first, second):
    gray1, _ = preprocess_frame(first)
    gray2, _ = preprocess_frame(second)
    return compute_motion_score(gray1, gray2)


def edges(first, second):
    _, blurred = preprocess_frame(first)
    return detect_edges(blurred)


def hough(first, second):
    return detect_lines(edges(first, second))


def motion_edges(first, second):
    motion(first, second)
    return edges(first, second)


CONFIGURATIONS = {
    "optical flow only": motion,
    "edges only": edges,
    "Hough lines only": hough,
    "optical flow + edges": motion_edges,
    "full RiskSight fusion": compute_hybrid_risk,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=Path("data/Drone.mp4"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=5)
    args = parser.parse_args(argv)
    frames = load_video_frames(args.video, max_frames=args.max_frames, resize_width=args.width)
    results = []
    for name, function in CONFIGURATIONS.items():
        print(f"Benchmarking {name} ...", flush=True)
        for first, second in list(zip(frames, frames[1:]))[:args.warmup_frames]:
            function(first, second)
        durations = []
        for first, second in zip(frames, frames[1:]):
            start = perf_counter()
            function(first, second)
            durations.append(perf_counter() - start)
        results.append({"configuration": name, **timing_summary(durations)})
    baseline = next(x["mean_ms"] for x in results if x["configuration"] == "full RiskSight fusion")
    for result in results:
        result["relative_cost"] = result["mean_ms"] / baseline
    payload = {"environment": machine_metadata(), **video_info(args.video, args.width), "warmup_frames": args.warmup_frames, "results": results}
    write_json("ablation.json", payload)
    write_csv("ablation.csv", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
