import unittest

from scripts.analyze_detections import iou, summarize_threshold


class DetectionAnalysisTest(unittest.TestCase):
    def test_iou_for_overlapping_boxes(self):
        self.assertAlmostEqual(
            iou((0, 0, 100, 100), (50, 50, 150, 150)),
            2500 / 17500,
            places=6
        )

    def test_threshold_summary_counts_tp_fp_fn(self):
        ground_truth = {
            "img1": [
                {"class_name": "with_mask", "box": (0, 0, 100, 100)},
                {"class_name": "without_mask", "box": (200, 200, 300, 300)},
            ]
        }
        detections = {
            "img1": [
                {"class_name": "with_mask", "score": 0.9, "box": (0, 0, 100, 100)},
                {"class_name": "without_mask", "score": 0.8, "box": (0, 0, 50, 50)},
            ]
        }

        rows = summarize_threshold(ground_truth, detections, threshold=0.5, min_iou=0.5)
        by_class = {row["class_name"]: row for row in rows}

        self.assertEqual(by_class["with_mask"]["tp"], 1)
        self.assertEqual(by_class["with_mask"]["fp"], 0)
        self.assertEqual(by_class["without_mask"]["tp"], 0)
        self.assertEqual(by_class["without_mask"]["fp"], 1)
        self.assertEqual(by_class["without_mask"]["fn"], 1)
        self.assertEqual(by_class["overall"]["tp"], 1)
        self.assertEqual(by_class["overall"]["fp"], 1)
        self.assertEqual(by_class["overall"]["fn"], 1)


if __name__ == "__main__":
    unittest.main()
