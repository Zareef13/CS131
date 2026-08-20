"""Detector-oriented evaluation against AI pseudo-labels, not human ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from risksight.pipeline import compute_hybrid_risk
from .common import ROOT, write_json


def load_pair(video: Path, index: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode frame t and t+1, preserving alignment with frame-t labels."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise ValueError(f"Could not open {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok1, first = capture.read(); ok2, second = capture.read()
    if not ok1 or not ok2:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0); first = second = None
        for current in range(index + 2):
            success, frame = capture.read()
            if not success: break
            if current == index: first = frame
            elif current == index + 1: second = frame
    capture.release()
    if first is None or second is None: raise ValueError(f"Could not decode pair at {index} from {video}")
    def resize(frame: np.ndarray) -> np.ndarray:
        height = int(frame.shape[0] * width / frame.shape[1])
        return cv2.cvtColor(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return resize(first), resize(second)


def boundary(mask: np.ndarray, width: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return mask.astype(bool) & ~(cv2.erode(mask.astype(np.uint8), kernel) > 0)


def tolerant_boundary_counts(prediction: np.ndarray, truth: np.ndarray, tolerance: int) -> dict[str, int]:
    pred_boundary, truth_boundary = boundary(prediction), boundary(truth)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    pred_near = cv2.dilate(pred_boundary.astype(np.uint8), kernel) > 0
    truth_near = cv2.dilate(truth_boundary.astype(np.uint8), kernel) > 0
    return {
        "matched_prediction_boundary": int(np.count_nonzero(pred_boundary & truth_near)),
        "prediction_boundary": int(np.count_nonzero(pred_boundary)),
        "matched_reference_boundary": int(np.count_nonzero(truth_boundary & pred_near)),
        "reference_boundary": int(np.count_nonzero(truth_boundary)),
    }


def boundary_metrics(counts: dict[str, int]) -> dict[str, float]:
    precision = counts["matched_prediction_boundary"] / counts["prediction_boundary"] if counts["prediction_boundary"] else 0.0
    recall = counts["matched_reference_boundary"] / counts["reference_boundary"] if counts["reference_boundary"] else 1.0
    return {"boundary_precision": precision, "boundary_recall": recall, "boundary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def average_precision(score: np.ndarray, target: np.ndarray) -> float:
    labels = target.astype(bool).ravel(); positives = int(labels.sum())
    if not positives: return 1.0
    order = np.argsort(-score.ravel(), kind="stable"); ranked = labels[order]
    cumulative = np.cumsum(ranked)
    return float(np.sum((cumulative / np.arange(1, len(ranked) + 1))[ranked]) / positives)


def load_frames(video: Path, annotations: Path, width: int) -> list[dict]:
    frames = []
    for record_path in sorted(annotations.glob("frame_*.json")):
        record = json.loads(record_path.read_text())
        if not record.get("usable_for_evaluation"): continue
        first, second = load_pair(video, int(record["frame_index"]), width)
        score = compute_hybrid_risk(first, second)["risk_display"]
        truth = cv2.imread(str(ROOT / record["mask"]), cv2.IMREAD_GRAYSCALE)
        truth = cv2.resize(truth, (score.shape[1], score.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        objects = []
        for item in record["obstacles"]:
            if not item.get("accepted"): continue
            object_mask = cv2.imread(str(ROOT / item["selected_mask_file"]), cv2.IMREAD_GRAYSCALE)
            objects.append(cv2.resize(object_mask, (score.shape[1], score.shape[0]), interpolation=cv2.INTER_NEAREST) > 0)
        frames.append({"index": record["frame_index"], "score": score, "truth": truth, "objects": objects})
    return frames


def evaluate(frames: list[dict], threshold: float, tolerance: int) -> dict:
    totals = {key: 0 for key in ("matched_prediction_boundary", "prediction_boundary", "matched_reference_boundary", "reference_boundary")}
    coverage_hits = {fraction: 0 for fraction in (.01, .05, .10, .25, .50, .75)}
    objects = frame_hits = 0; aps = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    for frame in frames:
        prediction = frame["score"] >= threshold
        counts = tolerant_boundary_counts(prediction, frame["truth"], tolerance)
        for key in totals: totals[key] += counts[key]
        aps.append(average_precision(frame["score"], cv2.dilate(boundary(frame["truth"]).astype(np.uint8), kernel) > 0))
        hit = False
        prediction_near = cv2.dilate(prediction.astype(np.uint8), kernel) > 0
        for object_mask in frame["objects"]:
            objects += 1; coverage = float(np.mean(prediction_near[object_mask])) if object_mask.any() else 0.0
            for fraction in coverage_hits: coverage_hits[fraction] += int(coverage >= fraction)
            hit |= coverage >= .01
        frame_hits += int(hit)
    return {"threshold": threshold, "tolerance_pixels": tolerance, "frames": len(frames), "objects": objects,
            **totals, **boundary_metrics(totals), "boundary_average_precision": float(np.mean(aps)) if aps else 0.0,
            **{f"object_recall_at_{int(fraction*100)}pct_coverage": coverage_hits[fraction] / objects if objects else 1.0 for fraction in coverage_hits},
            "frame_level_obstacle_recall": frame_hits / len(frames) if frames else 1.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path); parser.add_argument("annotations", type=Path)
    parser.add_argument("--width", type=int, default=640); parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument("--output-name", default="obstacle_detection_ai")
    args = parser.parse_args(argv)
    frames = load_frames(args.video, args.annotations, args.width)
    if len(frames) < 4: parser.error("at least four quality-approved frames are required for validation/test evaluation")
    validation, test = frames[::2], frames[1::2]
    candidates = [round(value / 100, 2) for value in range(5, 96, 5)]
    validation_rows = [evaluate(validation, threshold, args.tolerance) for threshold in candidates]
    selected = max(validation_rows, key=lambda row: (row["boundary_f1"], -abs(row["threshold"] - .5)))["threshold"]
    payload = {"reference_type": "AI-generated pseudo-ground truth; not human-verified", "task": "obstacle detection/localization, not semantic segmentation",
               "alignment": "mask frame t compared with RiskSight(frame t, frame t+1)", "split": "alternating temporal samples; threshold selected on validation only",
               "validation_indices": [f["index"] for f in validation], "test_indices": [f["index"] for f in test],
               "validation_threshold_sweep": validation_rows, "selected_threshold": selected,
               "held_out_test": evaluate(test, selected, args.tolerance), "all_frames_at_fixed_0_5": evaluate(frames, .5, args.tolerance)}
    write_json(f"{args.output_name}.json", payload); print(json.dumps(payload["held_out_test"], indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
