"""Safety tests for adapting frozen 8B-era entrypoint I/O to NF4 models."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import stage_shared as adapter


class InputSafetyTests(unittest.TestCase):
    def check(self, rows, development=None, confirmation=None):
        return adapter.validate_metadata_records(
            rows, development or {"development_a": 0, "development_b": 1},
            {"confirmation_a"} if confirmation is None else confirmation,
        )

    def test_development_rows_are_accepted(self):
        evidence = self.check([
            {"participant": "development_a", "fold": 0},
            {"participant": "development_b", "fold": 1},
        ])
        self.assertEqual(evidence, {"rows": 2, "participants": 2, "confirmation_rows": 0})

    def test_legitimate_missing_task_participants_are_allowed(self):
        # The global split has 372 participants, but some tasks have 371 or
        # 367 observed participants. A task need only be a non-empty subset.
        development = {f"development_{index}": index % 5 for index in range(372)}
        for observed_count in (367, 371):
            with self.subTest(observed_count=observed_count):
                rows = [
                    {"participant": participant, "fold": fold}
                    for participant, fold in list(development.items())[:observed_count]
                ]
                evidence = self.check(rows, development=development)
                self.assertEqual(evidence["participants"], observed_count)
                self.assertEqual(evidence["confirmation_rows"], 0)

    def test_confirmation_rows_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Non-development"):
            self.check([{"participant": "confirmation_a", "fold": 0}])

    def test_unknown_participants_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Non-development"):
            self.check([{"participant": "unknown", "fold": 0}])

    def test_fold_reassignment_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "reassignment"):
            self.check([{"participant": "development_a", "fold": 1}])

    def test_fractional_folds_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "reassignment"):
            self.check([{"participant": "development_a", "fold": 0.5}])

    def test_boolean_fold_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            self.check([{"participant": "development_a", "fold": False}])

    def test_overlap_is_rejected_before_reading_rows(self):
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            self.check([], confirmation={"development_a"})

    def test_empty_task_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "empty"):
            self.check([])


class PathAndSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.base = Path(self.temporary.name)
        self.snapshot, self.view = self.base / "snapshot", self.base / "view"
        self.snapshot.mkdir()
        self.view.mkdir()

    def tearDown(self):
        self.assertTrue(Path(self.temporary.name).resolve().is_relative_to(Path(__file__).resolve().parent))
        self.temporary.cleanup()

    def test_separate_view_is_allowed(self):
        adapter.validate_roots(self.snapshot, self.view)

    def test_source_equal_to_view_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            adapter.validate_roots(self.snapshot, self.snapshot)

    def test_view_inside_source_is_rejected(self):
        nested = self.snapshot / "newview"
        nested.mkdir()
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            adapter.validate_roots(self.snapshot, nested)

    def test_source_inside_view_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            adapter.validate_roots(self.snapshot, self.base)

    def test_explicit_nf4_names_have_separate_outputs(self):
        first = adapter.output_paths(self.view, adapter.MODELS[0], 60)
        second = adapter.output_paths(self.view, adapter.MODELS[1], 60)
        for key in ("selection", "shared", "control_cache", "control_result", "final", "cv"):
            self.assertNotEqual(first[key], second[key])
            self.assertIn(adapter.MODELS[0], str(first[key]))
        self.assertEqual(first["selection"].name, "selected_shared_centaur_70b_nf4_layer_60.json")
        self.assertIn("q512_rm0_unit0", str(first["cv"]))

    def test_legacy_model_alias_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            adapter.output_paths(self.view, "minitaur_8b", 60)

    def test_unextracted_layer_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            adapter.output_paths(self.view, adapter.MODELS[0], 24)

    def test_relative_output_escape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            adapter.checked_path(self.view, "../outside.json")

    def test_symlink_output_escape_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        try:
            (self.view / "results").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"This platform does not allow creating test symlinks: {error}")
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            adapter.output_paths(self.view, adapter.MODELS[0], 60)

    def frozen_fixture(self):
        inventory = {}
        for relative, text in (("configs/original/main.yaml", "unchanged-main\n"),
                               ("configs/original/splits.json", '{"split":"unchanged"}\n'),
                               ("scripts/original/frozen.py", "frozen_algorithm\n")):
            path = self.snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            inventory[relative] = adapter.sha256(path)
            if relative.startswith("configs/"):
                target = self.view / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        (self.snapshot / "source_inventory.json").write_text(json.dumps(inventory))

    def test_unchanged_source_and_config_pass(self):
        self.frozen_fixture()
        self.assertEqual(adapter.verify_sources(self.snapshot, self.view)["verified_source_files"], 3)

    def test_source_mutation_is_rejected(self):
        self.frozen_fixture()
        (self.snapshot / "scripts/original/frozen.py").write_text("changed_algorithm\n")
        with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
            adapter.verify_sources(self.snapshot, self.view)

    def test_view_config_mutation_is_rejected(self):
        self.frozen_fixture()
        (self.view / "configs/original/main.yaml").write_text("primary_layer_8b: 60\n")
        with self.assertRaisesRegex(RuntimeError, "unchanged frozen main"):
            adapter.verify_sources(self.snapshot, self.view)


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.view = Path(self.temporary.name)
        self.model = adapter.MODELS[0]
        self.paths = adapter.output_paths(self.view, self.model, 60)

    def tearDown(self):
        self.assertTrue(Path(self.temporary.name).resolve().is_relative_to(Path(__file__).resolve().parent))
        self.temporary.cleanup()

    def fake_controls(self):
        self.cache_writes = []
        original_finalize = lambda: "original_completeness_checks"
        module = SimpleNamespace(
            SPECS={"task_a": object()},
            _write_cache=lambda key, rows: self.cache_writes.append((key, rows)),
            shared_behavioral_rows=lambda selection: [{"model": "minitaur_8b", "validation_nll": .25}],
            null_intercept_rows=lambda *args: [{"model": "minitaur_8b", "validation_nll": .5}],
            finalize=original_finalize,
            fit_standard_controls=lambda value: value,
            fit_held_out_family=lambda: "original_family_transfer",
        )
        return module

    def test_rank_adapter_retains_frozen_condition_and_actual_model(self):
        original_main = lambda: None
        module = SimpleNamespace(main=original_main)
        adapter.configure_rank_module(module, self.view, self.paths, self.model, 60)
        args = module.parse_args()
        self.assertEqual((args.model, args.layer, args.pca_dim), (self.model, 60, 512))
        self.assertEqual((args.remove_pcs, args.unit_normalize, args.overwrite), (0, False, False))
        self.assertIs(module.main, original_main)
        self.assertEqual(module.MODEL_CHOICES, adapter.MODELS)

    def test_control_algorithm_functions_are_preserved(self):
        module = self.fake_controls()
        originals = [module.finalize, module.fit_standard_controls, module.fit_held_out_family]
        adapter.configure_control_module(module, self.view, self.paths, self.model)
        self.assertIs(module.finalize, originals[0])
        self.assertIs(module.fit_standard_controls, originals[1])
        self.assertIs(module.fit_held_out_family, originals[2])

    def test_hardcoded_row_identity_is_normalized_without_value_changes(self):
        module = self.fake_controls()
        adapter.configure_control_module(module, self.view, self.paths, self.model)
        module._write_cache("full_fold0", [{"model": "minitaur_8b", "validation_nll": .25}])
        self.assertEqual(self.cache_writes[0][1], [{"model": self.model, "validation_nll": .25}])
        self.assertEqual(module.shared_behavioral_rows({})[0]["model"], self.model)
        self.assertEqual(module.null_intercept_rows({}, {}, {})[0]["model"], self.model)

    def test_unsupported_row_identity_is_rejected(self):
        with self.assertRaises(RuntimeError):
            adapter.attributed_rows([], "minitaur_8b")

    def test_identity_normalization_does_not_mutate_input(self):
        original = [{"model": "minitaur_8b", "nll": .3}]
        output = adapter.attributed_rows(original, self.model)
        self.assertEqual(original[0]["model"], "minitaur_8b")
        self.assertEqual(output[0]["nll"], original[0]["nll"])

    def test_missing_matched_finetune_controls_are_explicit(self):
        module = self.fake_controls()
        adapter.configure_control_module(module, self.view, self.paths, self.model)
        rows = [{"control": "full_representation", "model": self.model}]
        module.add_finetune_controls(rows)
        self.assertEqual(len(rows), 1)
        unavailable = json.loads(self.paths["unavailable"].read_text())
        self.assertEqual(unavailable["state"], "dependency_unavailable")
        self.assertFalse(unavailable["8b_substitution_used"])
        self.assertFalse(unavailable["complete_original_control_battery"])

    def configured_data_fixture(self):
        module = self.fake_controls()
        adapter.configure_control_module(module, self.view, self.paths, self.model)
        self.paths["config"].parent.mkdir(parents=True)
        self.paths["config"].write_text("unchanged-frozen-config\n")
        selection = {
            "model": self.model, "layer": 60, "pca_dim": 512,
            "config_hash": adapter.sha256(self.paths["config"]),
            "remove_first_global_pcs": 0, "unit_length_normalization": False,
        }
        adapter.atomic_json(self.paths["selection"], selection)
        module.yaml = SimpleNamespace(safe_load=lambda value: {"config_text": value})
        self.load_calls = []

        def load(projected, states, tasks, dimension):
            self.load_calls.append((projected, states, tasks, dimension))
            return {"task_a": "unchanged_original_loader"}

        module.load_task_arrays = load
        return module, selection

    def test_control_data_loader_uses_actual_nf4_paths_and_original_dimension(self):
        module, selection = self.configured_data_fixture()
        _config, actual_selection, data = module._data()
        self.assertEqual(actual_selection, selection)
        self.assertEqual(data, {"task_a": "unchanged_original_loader"})
        self.assertEqual(self.load_calls, [(self.paths["projected"], self.paths["states"], ["task_a"], 512)])

    def test_control_data_rejects_other_model_selection(self):
        module, selection = self.configured_data_fixture()
        selection["model"] = adapter.MODELS[1]
        adapter.atomic_json(self.paths["selection"], selection)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            module._data()
        self.assertFalse(self.load_calls)

    def test_control_data_rejects_changed_configuration(self):
        module, _selection = self.configured_data_fixture()
        self.paths["config"].write_text("modified-config\n")
        with self.assertRaisesRegex(RuntimeError, "different configuration"):
            module._data()

    def test_control_data_rejects_sensitivity_preprocessing(self):
        module, selection = self.configured_data_fixture()
        selection["unit_length_normalization"] = True
        adapter.atomic_json(self.paths["selection"], selection)
        with self.assertRaisesRegex(RuntimeError, "primary preprocessing"):
            module._data()

    def test_nf4_manifests_never_claim_original_bf16_completion(self):
        path = self.paths["control_manifest"]
        path.parent.mkdir(parents=True)
        path.write_text('{"profile":"original","rows":7}')
        adapter.label_manifest(path, self.model, {"frozen_config_sha256": "pin"})
        payload = json.loads(path.read_text())
        self.assertEqual(payload["profile"], adapter.PROFILE)
        self.assertEqual(payload["model"], self.model)
        self.assertEqual(payload["rows"], 7)
        self.assertTrue(payload["core_sro_controls_complete"])
        self.assertFalse(payload["complete_original_control_battery"])
        self.assertFalse(payload["satisfies_original_exact_bf16_requirement"])


if __name__ == "__main__":
    unittest.main()
