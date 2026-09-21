import unittest

import numpy as np


class CspLdaMetricsTests(unittest.TestCase):
    def test_metrics_include_a_two_class_confusion_matrix(self) -> None:
        try:
            from training.train_csp_lda import metrics
        except ImportError:
            self.skipTest("scikit-learn is not installed")

        class Model:
            def predict(self, values):
                return np.asarray([0, 1, 1, 0])

        result = metrics(Model(), np.zeros((4, 2, 4)), np.asarray([0, 1, 0, 0]))
        self.assertEqual(result["epochs"], 4)
        self.assertEqual(result["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertGreaterEqual(result["balanced_accuracy"], 0.0)
