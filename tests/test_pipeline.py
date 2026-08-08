import unittest

import numpy as np

from risksight.pipeline import compute_hybrid_risk
from risksight.preprocessing import normalize_map


class PipelineTests(unittest.TestCase):
    def test_normalize_constant_map_is_zero(self) -> None:
        normalized = normalize_map(np.full((4, 4), 7, dtype=np.uint8))
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(np.count_nonzero(normalized), 0)

    def test_pipeline_returns_stable_shapes_and_ranges(self) -> None:
        frame1 = np.zeros((64, 96, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame1[16:48, 20:45] = 255
        frame2[16:48, 24:49] = 255

        result = compute_hybrid_risk(frame1, frame2)

        self.assertEqual(
            set(result),
            {
                "overlay",
                "risk_display",
                "motion_score",
                "edge_map",
                "line_map",
                "canny_edges",
            },
        )
        self.assertEqual(result["overlay"].shape, frame1.shape)
        self.assertEqual(result["overlay"].dtype, np.uint8)
        for key in ("risk_display", "motion_score", "edge_map", "line_map"):
            self.assertEqual(result[key].shape, frame1.shape[:2])
            self.assertGreaterEqual(float(result[key].min()), 0)
            self.assertLessEqual(float(result[key].max()), 1)


if __name__ == "__main__":
    unittest.main()
