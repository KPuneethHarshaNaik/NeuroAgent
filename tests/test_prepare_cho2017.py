import unittest

import numpy as np

from training.prepare_cho2017 import trials


class Cho2017Tests(unittest.TestCase):
    def test_event_windows_are_resampled(self) -> None:
        signal = np.zeros((2, 512 * 8), dtype=np.float32)
        events = np.zeros(signal.shape[1], dtype=np.uint8)
        events[1024] = 1
        output = trials(signal, events, 512, 160, 0.5, 3.5)
        self.assertEqual(output.shape, (1, 2, 480))


if __name__ == "__main__":
    unittest.main()
