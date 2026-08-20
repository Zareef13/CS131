import unittest

import numpy as np

from benchmarks.common import percentile, timing_summary
from benchmarks.evaluate_obstacle_detection import average_precision, boundary_metrics


class TimingTests(unittest.TestCase):
    def test_summary_and_percentiles(self) -> None:
        summary = timing_summary([0.001, 0.002, 0.003, 0.004])
        self.assertAlmostEqual(summary["mean_ms"], 2.5)
        self.assertAlmostEqual(summary["median_ms"], 2.5)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)
        self.assertAlmostEqual(summary["fps"], 400.0)
        self.assertEqual(percentile([7], 95), 7)

    def test_empty_timing_rejected(self) -> None:
        with self.assertRaises(ValueError):
            timing_summary([])


class DetectionMetricTests(unittest.TestCase):
    def test_average_precision_perfect_ranking(self) -> None:
        score = np.array([[0.9, 0.8], [0.2, 0.1]])
        target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        self.assertEqual(average_precision(score, target), 1.0)

    def test_boundary_metrics(self) -> None:
        counts = {"matched_prediction_boundary": 3, "prediction_boundary": 4, "matched_reference_boundary": 3, "reference_boundary": 6}
        result = boundary_metrics(counts)
        self.assertEqual(result["boundary_precision"], 0.75)
        self.assertEqual(result["boundary_recall"], 0.5)
        self.assertEqual(result["boundary_f1"], 0.6)


if __name__ == "__main__":
    unittest.main()
