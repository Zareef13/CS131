"""Generate and evaluate an isolated 50-frame drone pseudo-label study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from annotations.generate import main as generate
from .evaluate_obstacle_detection import main as evaluate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "annotations_ai_50"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="regenerate current caches; normally use restart-safe resume")
    parser.add_argument("--tolerance", type=int, default=10)
    args = parser.parse_args(argv)
    if not args.dry_run and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        parser.error("set GEMINI_API_KEY or GOOGLE_API_KEY")

    generation_args = ["--drone", "50", "--vehicle", "0", "--output-dir", str(OUTPUT)]
    generation_args.append("--force" if args.force else "--resume")
    if args.dry_run: generation_args.append("--dry-run")
    status = generate(generation_args)
    if status or args.dry_run: return status

    usable = []
    for record in sorted((OUTPUT / "drone").glob("frame_*.json")):
        import json
        payload = json.loads(record.read_text())
        if payload.get("usable_for_evaluation"): usable.append(payload["frame_index"])
    if len(usable) < 10:
        parser.error(f"only {len(usable)} frames passed quality gates; inspect annotations_ai_50/review_needed.json")

    return evaluate([
        "data/Drone.mp4", str(OUTPUT / "drone"), "--width", "640",
        "--tolerance", str(args.tolerance), "--output-name", "obstacle_detection_drone_ai_50",
    ])


if __name__ == "__main__": raise SystemExit(main())
