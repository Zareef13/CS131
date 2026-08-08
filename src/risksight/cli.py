"""Command-line entry point for RiskSight."""

import argparse
from pathlib import Path

from .video import load_video_frames

DEFAULT_VIDEOS = (Path("data/car.MP4"), Path("data/Drone.mp4"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate classical-CV obstacle-awareness overlays from video."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        default=list(DEFAULT_VIDEOS),
        help="input video path(s); defaults to the two project samples",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--max-frames", type=int, help="limit decoded frames for a quick run"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .visualization import save_all_outputs

    for video_path in args.videos:
        print(f"Processing: {video_path}")
        try:
            frames = load_video_frames(video_path, max_frames=args.max_frames)
            print(f"Loaded frames: {len(frames)}")
            print(f"Frame shape: {frames[0].shape}")
            save_all_outputs(frames, args.output_dir, video_path.stem.lower())
        except (FileNotFoundError, ValueError) as error:
            print(f"Error: {error}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
