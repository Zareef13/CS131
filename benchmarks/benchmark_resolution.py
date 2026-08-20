"""Measure RiskSight scaling at several aspect-ratio-preserving widths."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from risksight.pipeline import compute_hybrid_risk
from risksight.video import load_video_frames

from .common import machine_metadata, timing_summary, video_info, warm_up, write_csv, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=Path("data/Drone.mp4"))
    parser.add_argument("--widths", nargs="+", type=int, default=[320, 480, 640, 960])
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=5)
    args = parser.parse_args(argv)
    results = []
    for width in args.widths:
        print(f"Benchmarking width {width} ...", flush=True)
        frames = load_video_frames(args.video, max_frames=args.max_frames, resize_width=width)
        warm_up(frames, compute_hybrid_risk, args.warmup_frames)
        durations = []
        for first, second in zip(frames, frames[1:]):
            start = perf_counter()
            compute_hybrid_risk(first, second)
            durations.append(perf_counter() - start)
        results.append({"width": width, **timing_summary(durations)})
    payload = {"environment": machine_metadata(), **video_info(args.video, 640), "warmup_frames": args.warmup_frames, "results": results}
    write_json("resolution.json", payload)
    write_csv("resolution.csv", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
