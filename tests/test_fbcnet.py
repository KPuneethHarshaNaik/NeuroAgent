import unittest

import torch
import numpy as np

from training.fbcnet import FBCNet
from training.train_fbcnet import normalize_epochs, task_data


class FBCNetShapeTests(unittest.TestCase):
    def test_supports_all_action_class_count(self) -> None:
        output = FBCNet(channels=64, samples=481, classes=8)(torch.zeros((2, 1, 64, 481)))
        self.assertEqual(tuple(output.shape), (2, 8))

    def test_hand_side_task_drops_feet_and_remaps_indices(self) -> None:
        epochs = np.zeros((8, 2, 4), dtype=np.float32)
        labels = np.arange(8)
        selected_epochs, selected_labels, splits, names = task_data(epochs, labels, {"train": [0, 1, 4], "validation": [2, 3], "test": [5, 6, 7]}, "hand_side", {})
        self.assertEqual(selected_epochs.shape[0], 4)
        self.assertEqual(selected_labels.tolist(), [0, 1, 0, 1])
        self.assertEqual(splits, {"train": [0, 1], "validation": [2, 3], "test": []})
        self.assertEqual(names, {0: "left_hand", 1: "right_hand"})

    def test_epoch_channel_normalization_does_not_use_other_epochs(self) -> None:
        epochs = np.array([[[1.0, 3.0]], [[10.0, 14.0]]], dtype=np.float32)
        normalized, mean, std = normalize_epochs(epochs, [0], "epoch_channel")
        np.testing.assert_allclose(normalized[0], [[-1.0, 1.0]])
        np.testing.assert_allclose(normalized[1], [[-1.0, 1.0]])
        self.assertIsNone(mean)
        self.assertIsNone(std)


if __name__ == "__main__":
    unittest.main()
