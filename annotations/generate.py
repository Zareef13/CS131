"""Generate collision-relevant Gemini + local SAM 2 pseudo-annotations."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import REFERENCE_LABEL
from .audit import save_audits
from .gemini import MODEL as GEMINI_MODEL, PROMPT, locate
from .geometry import merge_masks, normalized_box_to_pixels, normalized_point_to_pixels, rejection_reasons
from .sam import MODEL as SAM_MODEL, Sam2Segmenter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "annotations_ai"
GENERATION_SCHEMA_VERSION = 3


def current_gemini_cache(path: Path, model: str) -> bool:
    """Return true only for a cache produced by the current prompt/schema."""
    if not path.exists(): return False
    try:
        record = json.loads(path.read_text()); payload = record["payload"]
        return (record.get("model") == model and record.get("prompt") == PROMPT
                and isinstance(payload.get("frame_usable_for_annotation"), bool)
                and isinstance(payload.get("exclusion_reason"), str)
                and all("forward_path_reason" in item for item in payload.get("obstacles", [])))
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def current_output_record(path: Path) -> bool:
    if not path.exists(): return False
    try:
        record = json.loads(path.read_text())
        return all(key in record for key in ("frame_usable_for_annotation", "usable_for_evaluation", "review_reasons"))
    except json.JSONDecodeError:
        return False


def decodable_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video)); count = 0
    if not capture.isOpened(): raise ValueError(f"Could not open {video}")
    while capture.grab(): count += 1
    capture.release(); return count


def sampled_indices(total: int, requested: int) -> list[int]:
    if requested < 0 or total < 2: raise ValueError("invalid sample request")
    return sorted(set(int(x) for x in np.linspace(0, total - 2, min(requested, total - 1))))


def extract(video: Path, domain: str, indices: list[int]) -> dict[int, Path]:
    directory = OUTPUT / domain; directory.mkdir(parents=True, exist_ok=True)
    needed = set(indices); found = {}; capture = cv2.VideoCapture(str(video)); index = 0
    while needed - found.keys():
        ok, frame = capture.read()
        if not ok: break
        if index in needed:
            path = directory / f"frame_{index:04d}.png"; cv2.imwrite(str(path), frame); found[index] = path
        index += 1
    capture.release()
    if set(found) != needed: raise ValueError(f"Could not extract {sorted(needed-set(found))}")
    return found


def gemini_copy(frame_bgr: np.ndarray, max_dimension: int = 512) -> bytes:
    scale = min(1.0, max_dimension / max(frame_bgr.shape[:2]))
    resized = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok: raise ValueError("could not encode Gemini input")
    return encoded.tobytes()


def grouped_split(records: list[dict[str, Any]], group_size: int = 5) -> dict[str, list[str]]:
    result = {"validation": [], "test": []}
    for domain in sorted({record["domain"] for record in records}):
        ordered = sorted((record for record in records if record["domain"] == domain and record.get("usable_for_evaluation", True)), key=lambda x: x["frame_index"])
        groups = [ordered[i:i + group_size] for i in range(0, len(ordered), group_size)]
        validation_groups = max(1, round(len(groups) * 0.20)) if groups else 0
        for group_index, group in enumerate(groups):
            split = "validation" if group_index < validation_groups else "test"
            result[split].extend(record["mask"] for record in group)
    return result


def select_valid_candidate(masks: np.ndarray, scores: list[float], box: list[int], confidence: float, minimum_confidence: float) -> tuple[int, list[str]]:
    """Prefer SAM's highest-scored candidate that passes deterministic checks."""
    ordered = sorted(range(len(scores)), key=lambda candidate: scores[candidate], reverse=True)
    checked = [(candidate, rejection_reasons(masks[candidate], box, confidence, minimum_confidence)) for candidate in ordered]
    return next(((candidate, reasons) for candidate, reasons in checked if not reasons), checked[0])


def main(argv: list[str] | None = None) -> int:
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", type=int, default=10); parser.add_argument("--vehicle", type=int, default=5)
    parser.add_argument("--annotator", choices=("gemini-sam",), default="gemini-sam")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--force", action="store_true")
    parser.add_argument("--gemini-model", default=GEMINI_MODEL); parser.add_argument("--sam-model", default=SAM_MODEL)
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="isolated annotation output directory")
    args = parser.parse_args(argv)
    requested_output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    OUTPUT = requested_output.resolve()
    try: OUTPUT.relative_to(ROOT)
    except ValueError: parser.error("--output-dir must be inside the project directory")
    if args.force and args.resume: parser.error("--force and --resume are mutually exclusive")
    if "pro" in args.gemini_model.lower(): parser.error("Pro-tier Gemini models are not permitted for this annotation pipeline")
    specs = [("drone", Path("data/Drone.mp4"), args.drone), ("vehicle", Path("data/car.MP4"), args.vehicle)]
    jobs = [(domain, video, index) for domain, video, count in specs if count > 0 for index in sampled_indices(decodable_count(video), count)]
    if args.limit is not None: jobs = jobs[:args.limit]
    cache_hits = sum(current_gemini_cache(OUTPUT / "cache" / "gemini" / domain / f"frame_{index:04d}.json", args.gemini_model) for domain, _, index in jobs)
    estimated_calls = len(jobs) if args.force else len(jobs) - cache_hits
    print(f"Frames: {len(jobs)}; estimated paid Gemini calls: {estimated_calls}; model: {args.gemini_model}")
    print(f"Local SAM model: {args.sam_model} (CPU); full 300-frame command is supported but not started.")
    if args.dry_run: return 0
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key: parser.error("set GEMINI_API_KEY or GOOGLE_API_KEY")
    from google import genai
    client = genai.Client(api_key=api_key); segmenter = Sam2Segmenter(args.sam_model, "cpu")
    extracted: dict[tuple[str, int], Path] = {}
    for domain, video, _ in specs:
        domain_indices = [index for job_domain, _, index in jobs if job_domain == domain]
        for index, path in extract(video, domain, domain_indices).items(): extracted[(domain, index)] = path
    records = []; review = []; gemini_latencies = []; sam_latencies = []; costs = []
    for domain, _, index in jobs:
        output_json = OUTPUT / domain / f"frame_{index:04d}.json"
        gemini_cache = OUTPUT / "cache" / "gemini" / domain / f"frame_{index:04d}.json"
        # A previous interrupted --force can leave the output record while
        # having already removed its cache. Such a frame is incomplete and
        # must be regenerated instead of being skipped.
        cache_is_current = current_gemini_cache(gemini_cache, args.gemini_model)
        if current_output_record(output_json) and cache_is_current and args.resume and not args.force:
            records.append(json.loads(output_json.read_text())); continue
        frame_path = extracted[(domain, index)]; frame_bgr = cv2.imread(str(frame_path)); frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        semantic_changed = args.force or not cache_is_current
        if semantic_changed and gemini_cache.exists(): gemini_cache.unlink()
        if semantic_changed:
            for stale_sam_cache in (OUTPUT / "cache" / "sam" / domain).glob(f"frame_{index:04d}_object_*.npz"): stale_sam_cache.unlink()
            for stale_object_mask in (OUTPUT / domain).glob(f"frame_{index:04d}_object_*.png"): stale_object_mask.unlink()
        semantic = locate(client, gemini_copy(frame_bgr), gemini_cache, args.gemini_model)
        gemini_latencies.append(semantic.latency_seconds); costs.append(semantic.cost_usd)
        accepted = []; obstacles = []
        for obstacle_index, obstacle in enumerate(semantic.payload["obstacles"]):
            box = normalized_box_to_pixels(obstacle["box"], frame_rgb.shape[1], frame_rgb.shape[0])
            point = normalized_point_to_pixels(obstacle["point"], frame_rgb.shape[1], frame_rgb.shape[0])
            sam_cache = OUTPUT / "cache" / "sam" / domain / f"frame_{index:04d}_object_{obstacle_index}.npz"
            if args.force and sam_cache.exists(): sam_cache.unlink()
            sam = segmenter.segment(frame_rgb, box, point, sam_cache); sam_latencies.append(sam.latency_seconds)
            selected_index, reasons = select_valid_candidate(sam.masks, sam.scores, box, float(obstacle["confidence"]), args.minimum_confidence)
            mask = sam.masks[selected_index]
            selected_path = OUTPUT / domain / f"frame_{index:04d}_object_{obstacle_index}.png"
            cv2.imwrite(str(selected_path), mask.astype(np.uint8) * 255)
            accepted_status = not reasons
            if accepted_status: accepted.append(mask)
            item = {**obstacle, "box_pixels": box, "point_pixels": point, "sam_candidate_scores": sam.scores, "sam_model_top_index": sam.selected_index, "sam_selected_index": selected_index, "sam_mask_score": sam.scores[selected_index], "sam_inference_seconds": sam.latency_seconds, "selected_mask_file": str(selected_path.relative_to(ROOT)), "accepted": accepted_status, "rejection_reasons": reasons, "mask_area_pixels": int(mask.sum()), "mask_area_fraction": float(mask.mean())}
            obstacles.append(item)
        merged = merge_masks(accepted, frame_rgb.shape[:2]); mask_path = OUTPUT / domain / f"frame_{index:04d}_mask.png"; cv2.imwrite(str(mask_path), merged)
        area_fraction = float((merged > 0).mean())
        frame_reasons = [reason for item in obstacles for reason in item["rejection_reasons"]]
        if area_fraction > 0.50: frame_reasons.append("merged_mask_over_50_percent")
        frame_usable = semantic.payload["frame_usable_for_annotation"]
        if not frame_usable: frame_reasons.append("semantic_frame_excluded")
        flagged = bool(frame_reasons)
        record = {"reference_type": REFERENCE_LABEL, "domain": domain, "frame_index": index, "source_frame": str(frame_path.relative_to(ROOT)), "mask": str(mask_path.relative_to(ROOT)), "gemini_model": args.gemini_model, "sam_model": args.sam_model, "sam_device": "cpu", "frame_usable_for_annotation": frame_usable, "exclusion_reason": semantic.payload["exclusion_reason"], "frame_has_relevant_obstacle": semantic.payload["frame_has_relevant_obstacle"], "obstacles": obstacles, "valid_mask": frame_usable and (not semantic.payload["frame_has_relevant_obstacle"] or bool(accepted)), "usable_for_evaluation": frame_usable and not flagged, "flagged_for_review": flagged, "review_reasons": frame_reasons, "final_mask_area_fraction": area_fraction}
        output_json.write_text(json.dumps(record, indent=2) + "\n"); records.append(record)
        if record["flagged_for_review"]: review.append({"domain": domain, "frame_index": index, "annotation": str(output_json.relative_to(ROOT)), "exclusion_reason": record["exclusion_reason"], "reasons": frame_reasons})
    review = [{"domain": record["domain"], "frame_index": record["frame_index"], "annotation": str((OUTPUT / record["domain"] / f"frame_{record['frame_index']:04d}.json").relative_to(ROOT)), "exclusion_reason": record.get("exclusion_reason", ""), "reasons": record.get("review_reasons", [])} for record in records if record.get("flagged_for_review")]
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "review_needed.json").write_text(json.dumps(review, indent=2) + "\n")
    (OUTPUT / "splits.json").write_text(json.dumps({"method": "sorted temporal groups of five; first 20% of groups validation", **grouped_split(records)}, indent=2) + "\n")
    cache_paths = [OUTPUT / "cache" / "gemini" / r["domain"] / f"frame_{r['frame_index']:04d}.json" for r in records]
    cache_records = [json.loads(path.read_text()) for path in cache_paths if path.exists()]
    total_cost = sum(item.get("cost_usd", 0.0) for item in cache_records)
    all_obstacles = [item for record in records for item in record["obstacles"]]
    all_sam_times = [item["sam_inference_seconds"] for item in all_obstacles]
    metadata = {"schema_version": 2, "reference_type": REFERENCE_LABEL, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "gemini_model": args.gemini_model, "gemini_prompt": PROMPT, "gemini_max_dimension": 512, "sam_model": args.sam_model, "sam_device": "cpu", "sam_checkpoint_documented_size_mb": 156, "pilot_frames": len(records), "gemini_api_calls_this_invocation": estimated_calls, "gemini_total_cost_usd": total_cost, "average_cost_per_frame_usd": total_cost / len(records) if records else 0, "average_gemini_latency_seconds": float(np.mean([item.get("latency_seconds", 0.0) for item in cache_records])) if cache_records else 0, "average_sam_inference_seconds": float(np.mean(all_sam_times)) if all_sam_times else 0, "individual_masks_passing_percent": 100 * sum(item["accepted"] for item in all_obstacles) / len(all_obstacles) if all_obstacles else 100.0, "frames_with_valid_masks_percent": 100 * sum(r["valid_mask"] for r in records) / len(records), "frames_flagged_percent": 100 * sum(r["flagged_for_review"] for r in records) / len(records), "flagged_frame_count": sum(r["flagged_for_review"] for r in records), "average_relevant_obstacles": float(np.mean([len(r["obstacles"]) for r in records])), "average_final_mask_area_fraction": float(np.mean([r["final_mask_area_fraction"] for r in records])), "records": [str((OUTPUT / r["domain"] / f"frame_{r['frame_index']:04d}.json").relative_to(ROOT)) for r in records]}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_audits(records, ROOT, ROOT / "outputs" / "annotation_audit")
    print(json.dumps(metadata, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
