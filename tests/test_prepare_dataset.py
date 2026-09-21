import unittest

from training.prepare_dataset import subject_splits


class SubjectSplitTests(unittest.TestCase):
    def test_subjects_are_disjoint_and_complete(self) -> None:
        subjects = [f"S{value:03d}" for value in range(1, 110)]
        splits = subject_splits(subjects)
        assigned = splits["train"] + splits["validation"] + splits["test"]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(sorted(assigned), subjects)
        self.assertEqual((len(splits["train"]), len(splits["validation"]), len(splits["test"])), (76, 17, 16))


if __name__ == "__main__":
    unittest.main()
