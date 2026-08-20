import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from annotations.gemini import locate, validate_response
from annotations.generate import select_valid_candidate
from annotations.geometry import merge_masks, normalized_box_to_pixels, normalized_point_to_pixels, rejection_reasons


VALID = {"frame_usable_for_annotation": True, "exclusion_reason": "", "frame_has_relevant_obstacle": True, "obstacles": [{"label": "pillar", "relevance": "high", "forward_path_reason": "blocks corridor", "box": [0.1, 0.2, 0.6, 0.8], "point": [0.3, 0.4], "confidence": 0.9}]}


class GeometryTests(unittest.TestCase):
    def test_coordinate_conversion(self):
        self.assertEqual(normalized_box_to_pixels([0, 0, 1, 1], 101, 51), [0, 0, 100, 50])
        self.assertEqual(normalized_point_to_pixels([0.5, 0.5], 101, 51), [50, 25])

    def test_merge_masks(self):
        first = np.array([[1, 0], [0, 0]], dtype=bool); second = np.array([[0, 0], [0, 1]], dtype=bool)
        merged = merge_masks([first, second], (2, 2))
        np.testing.assert_array_equal(merged, np.array([[255, 0], [0, 255]], dtype=np.uint8))

    def test_empty_oversized_tiny_and_low_confidence_rejections(self):
        self.assertIn("empty_mask", rejection_reasons(np.zeros((10, 10)), [1, 1, 8, 8], 0.9))
        self.assertIn("mask_over_70_percent", rejection_reasons(np.ones((10, 10)), [0, 0, 9, 9], 0.9))
        tiny = np.zeros((100, 100)); tiny[50, 50] = 1
        self.assertIn("mask_tiny_relative_to_box", rejection_reasons(tiny, [10, 10, 90, 90], 0.9))
        self.assertIn("gemini_confidence_below_threshold", rejection_reasons(tiny, [10, 10, 90, 90], 0.1))

    def test_selects_lower_scored_valid_sam_candidate(self):
        oversized = np.ones((100, 100), dtype=bool)
        valid = np.zeros((100, 100), dtype=bool); valid[20:80, 20:80] = True
        index, reasons = select_valid_candidate(np.array([oversized, valid]), [0.9, 0.8], [10, 10, 90, 90], 0.9, 0.5)
        self.assertEqual(index, 1); self.assertEqual(reasons, [])


class GeminiTests(unittest.TestCase):
    def test_malformed_json_rejected(self):
        with self.assertRaises(ValueError): validate_response({"obstacles": []})

    def test_common_zero_to_thousand_coordinates_are_normalized(self):
        payload = {"frame_usable_for_annotation": True, "exclusion_reason": "", "frame_has_relevant_obstacle": True, "obstacles": [{"label": "wall", "relevance": "high", "forward_path_reason": "blocks corridor", "box": [100, 200, 600, 800], "point": [300, 400], "confidence": 94}]}
        result = validate_response(payload)
        self.assertEqual(result["obstacles"][0]["box"], [0.1, 0.2, 0.6, 0.8])
        self.assertEqual(result["obstacles"][0]["confidence"], 0.94)
        self.assertEqual(len(result["normalization_warnings"]), 2)

    def test_unusable_frame_requires_reason_and_no_obstacles(self):
        payload = {"frame_usable_for_annotation": False, "exclusion_reason": "pause menu", "frame_has_relevant_obstacle": False, "obstacles": []}
        self.assertEqual(validate_response(payload), payload)

    def test_usable_none_reason_and_reversed_bounds_are_repaired(self):
        payload = {**VALID, "exclusion_reason": "none", "obstacles": [{**VALID["obstacles"][0], "box": [0.1, 0.9, 0.4, 0.7], "point": [0.2, 0.8]}]}
        result = validate_response(payload)
        self.assertEqual(result["exclusion_reason"], "")
        self.assertEqual(result["obstacles"][0]["box"], [0.1, 0.7, 0.4, 0.9])

    def test_outside_point_is_replaced_with_box_center(self):
        payload = {**VALID, "obstacles": [{**VALID["obstacles"][0], "point": [0.9, 0.9]}]}
        result = validate_response(payload)
        self.assertEqual(result["obstacles"][0]["point"], [0.35, 0.5])

    def test_mocked_response_and_metadata_serialization(self):
        response = SimpleNamespace(text=json.dumps(VALID), usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            result = locate(None, b"jpeg", path, request=lambda **kwargs: response)
            self.assertEqual(result.payload, VALID); self.assertTrue(path.exists())
            serialized = json.loads(path.read_text())
            self.assertEqual(serialized["model"], "gemini-3.1-flash-lite")


if __name__ == "__main__": unittest.main()
