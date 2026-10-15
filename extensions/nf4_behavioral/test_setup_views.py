import copy
from pathlib import Path
import tempfile
import unittest
import setup_views as setup


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.splits = {"development": [{"participant": f"d{i}", "fold": i % 5} for i in range(372)],
                       "confirmation": [f"c{i}" for i in range(150)]}
        self.development, self.confirmation = setup.split_contract(self.splits)

    def test_development_exact_membership_and_folds(self):
        setup.validate_participants(list(self.development), list(self.development.values()),
                                    "development", self.development, self.confirmation)

    def test_confirmation_never_enters_development(self):
        people = list(self.development)
        people[0] = "c0"
        with self.assertRaises(ValueError):
            setup.validate_participants(people, list(self.development.values()), "development",
                                        self.development, self.confirmation)

    def test_wrong_development_fold_rejected(self):
        folds = list(self.development.values())
        folds[0] = 4
        with self.assertRaises(ValueError):
            setup.validate_participants(list(self.development), folds, "development",
                                        self.development, self.confirmation)

    def test_development_never_enters_confirmation(self):
        with self.assertRaises(ValueError):
            setup.validate_participants(["d0"], [-1], "confirmation_t1",
                                        self.development, self.confirmation)

    def test_partial_confirmation_task_is_allowed(self):
        setup.validate_participants(["c0", "c1"], [-1, -1], "confirmation_t2",
                                    self.development, self.confirmation)

    def test_legitimate_partial_development_task_is_allowed(self):
        setup.validate_participants(["d0", "d1"], [0, 1], "development",
                                    self.development, self.confirmation)

    def test_empty_development_rejected(self):
        with self.assertRaises(ValueError):
            setup.validate_participants([], [], "development", self.development, self.confirmation)

    def test_metadata_identity_and_row_alignment(self):
        frame = {"participant": ["d0", "d1"], "model": ["centaur_70b_nf4"] * 2,
                 "split": ["development"] * 2, "task": ["task"] * 2, "state_row": [0, 1]}
        setup.validate_identity(frame, "centaur_70b_nf4", "development", "task", {"rows": 2, "sessions": 2})
        frame["state_row"] = [1, 0]
        with self.assertRaises(ValueError):
            setup.validate_identity(frame, "centaur_70b_nf4", "development", "task", {"rows": 2, "sessions": 2})

    def test_wrong_model_metadata_rejected(self):
        frame = {"participant": ["d0"], "model": ["llama_base_70b_nf4"],
                 "split": ["development"], "task": ["task"], "state_row": [0]}
        with self.assertRaises(ValueError):
            setup.validate_identity(frame, "centaur_70b_nf4", "development", "task", {"rows": 1, "sessions": 1})

    def test_duplicate_split_members_rejected(self):
        splits = copy.deepcopy(self.splits)
        splits["development"][-1] = splits["development"][0]
        with self.assertRaises(ValueError):
            setup.split_contract(splits)

    def test_external_write_path_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            with self.assertRaises(ValueError):
                setup.bounded(Path(temporary).parent / "outside", temporary)

    def test_legacy_model_names_are_not_used(self):
        self.assertEqual(set(setup.MODELS), {"centaur_70b_nf4", "llama_base_70b_nf4"})

    def test_parent_traversal_write_path_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            with self.assertRaises(ValueError):
                setup.bounded(Path(temporary) / ".." / "outside", temporary)


if __name__ == "__main__":
    unittest.main()
