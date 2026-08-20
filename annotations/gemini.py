"""Paid semantic localization stage; never requests segmentation polygons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


MODEL = "gemini-3.1-flash-lite"
INPUT_USD_PER_MILLION = 0.25
OUTPUT_USD_PER_MILLION = 1.50
PROMPT = """Annotate a frame only if it is a usable forward-facing FPV navigation view. Mark it unusable if a pause/menu/map/HUD panel obscures the scene, the camera is an exterior side view of a vehicle, a passenger/window view, a static scenic shot, or otherwise not a forward navigation view. For an unusable frame return no obstacles and give a short exclusion reason.

For a usable frame, identify at most four visible solid obstacles that intersect or materially narrow the drone's current or near-future forward flight corridor. Include walls, pillars, bridge supports, directly blocking building faces, vehicles, poles, trunks, large branches, barriers, fences, beams, gates, and large rocks only when they are actually in that corridor. Exclude sky, water unless descending into it, shadows, reflections, texture boundaries, distant/off-path scenery, whole side/background buildings, open ground, roads, grass, and HUD overlays. Do not select an object merely because it has edges or motion. Give a short forward-path reason for each object. Return structured JSON only. Coordinates must be decimal fractions from 0.0 to 1.0 (never 0 to 1000): box [x_min,y_min,x_max,y_max] and positive point [x,y] inside the object. Do not return polygons or reasoning beyond the requested short fields."""
SCHEMA = {
    "type": "object",
    "properties": {
        "frame_usable_for_annotation": {"type": "boolean"},
        "exclusion_reason": {"type": "string"},
        "frame_has_relevant_obstacle": {"type": "boolean"},
        "obstacles": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "label": {"type": "string"},
            "relevance": {"type": "string", "enum": ["low", "medium", "high"]},
            "forward_path_reason": {"type": "string"},
            "box": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 4, "maxItems": 4},
            "point": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 2, "maxItems": 2},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }, "required": ["label", "relevance", "forward_path_reason", "box", "point", "confidence"]}},
    },
    "required": ["frame_usable_for_annotation", "exclusion_reason", "frame_has_relevant_obstacle", "obstacles"],
}


def validate_response(payload: Any) -> dict[str, Any]:
    if (not isinstance(payload, dict) or not isinstance(payload.get("frame_usable_for_annotation"), bool)
            or not isinstance(payload.get("exclusion_reason"), str)
            or not isinstance(payload.get("frame_has_relevant_obstacle"), bool)
            or not isinstance(payload.get("obstacles"), list)):
        raise ValueError("malformed Gemini response")
    warnings: list[str] = []
    if len(payload["obstacles"]) > 4:
        raise ValueError("Gemini returned more than four obstacles")
    if not payload["frame_usable_for_annotation"]:
        if payload["frame_has_relevant_obstacle"] or payload["obstacles"] or not payload["exclusion_reason"].strip():
            raise ValueError("unusable Gemini frame response is internally inconsistent")
    elif payload["exclusion_reason"].strip():
        # Models often serialize the requested empty value as "none". It has
        # no semantic effect on a usable frame, so normalize instead of paying
        # for another identical request.
        warnings.append("usable_frame:cleared_exclusion_reason")
        payload["exclusion_reason"] = ""
    for obstacle_index, obstacle in enumerate(payload["obstacles"]):
        if not isinstance(obstacle, dict) or not all(key in obstacle for key in ("label", "relevance", "forward_path_reason", "box", "point", "confidence")):
            raise ValueError("malformed Gemini obstacle")
        box, point = obstacle["box"], obstacle["point"]
        if len(box) != 4 or len(point) != 2 or any(not isinstance(x, (int, float)) for x in [*box, *point]):
            raise ValueError("invalid Gemini coordinates")
        coordinates = [float(x) for x in [*box, *point]]
        if any(x > 1 for x in coordinates):
            # Gemini object-localization training commonly emits 0..1000 even
            # when prompted for fractions. Repair only that unmistakable scale.
            if min(coordinates) >= 0 and 10 <= max(coordinates) <= 1000:
                obstacle["box"] = [float(x) / 1000 for x in box]
                obstacle["point"] = [float(x) / 1000 for x in point]
                box, point = obstacle["box"], obstacle["point"]
                warnings.append(f"obstacle_{obstacle_index}:converted_coordinates_0_1000_to_0_1")
            else:
                raise ValueError("invalid Gemini coordinates")
        if any(not 0 <= x <= 1 for x in [*box, *point]):
            raise ValueError("invalid Gemini coordinates")
        if isinstance(obstacle["confidence"], (int, float)) and 1 < obstacle["confidence"] <= 100:
            obstacle["confidence"] = float(obstacle["confidence"]) / 100
            warnings.append(f"obstacle_{obstacle_index}:converted_confidence_percent_to_fraction")
        if box[2] <= box[0] or box[3] <= box[1]:
            repaired = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
            if repaired[2] > repaired[0] and repaired[3] > repaired[1]:
                obstacle["box"] = repaired
                box = repaired
                warnings.append(f"obstacle_{obstacle_index}:sorted_reversed_box_bounds")
            else:
                raise ValueError("invalid Gemini box or confidence")
        if not 0 <= obstacle["confidence"] <= 1:
            raise ValueError("invalid Gemini box or confidence")
        if not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
            # The box is the primary localization signal. A center point is a
            # valid deterministic SAM prompt and avoids paying for a retry.
            obstacle["point"] = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            warnings.append(f"obstacle_{obstacle_index}:replaced_outside_point_with_box_center")
    if not payload["frame_has_relevant_obstacle"] and payload["obstacles"]:
        raise ValueError("Gemini response is internally inconsistent")
    if warnings:
        payload["normalization_warnings"] = warnings
    return payload


@dataclass
class GeminiResult:
    payload: dict[str, Any]
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool


def locate(client: Any, image_bytes: bytes, cache_path: Path, model: str = MODEL, request: Callable[..., Any] | None = None) -> GeminiResult:
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return GeminiResult(validate_response(cached["payload"]), cached.get("latency_seconds", 0.0), cached.get("input_tokens", 0), cached.get("output_tokens", 0), cached.get("cost_usd", 0.0), True)
    from google.genai import types
    request = request or client.models.generate_content
    last_error: Exception | None = None
    for _ in range(3):
        start = perf_counter()
        try:
            response = request(
                model=model,
                contents=[PROMPT, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
                config={"response_mime_type": "application/json", "response_json_schema": SCHEMA, "thinking_config": {"thinking_level": "minimal"}},
            )
            latency = perf_counter() - start
            payload = validate_response(json.loads(response.text))
            usage = getattr(response, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            cost = input_tokens * INPUT_USD_PER_MILLION / 1_000_000 + output_tokens * OUTPUT_USD_PER_MILLION / 1_000_000
            record = {"payload": payload, "model": model, "prompt": PROMPT, "latency_seconds": latency, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost}
            cache_path.parent.mkdir(parents=True, exist_ok=True); cache_path.write_text(json.dumps(record, indent=2) + "\n")
            return GeminiResult(payload, latency, input_tokens, output_tokens, cost, False)
        except (ValueError, json.JSONDecodeError) as error:
            last_error = error
            malformed_path = cache_path.with_name(cache_path.stem + ".malformed.json")
            malformed_path.parent.mkdir(parents=True, exist_ok=True)
            previous = json.loads(malformed_path.read_text()) if malformed_path.exists() else []
            previous.append({"attempt": len(previous) + 1, "error": str(error), "response_text": getattr(locals().get("response", None), "text", None)})
            malformed_path.write_text(json.dumps(previous, indent=2) + "\n")
    raise ValueError(f"Gemini returned malformed JSON after three attempts: {last_error}")
