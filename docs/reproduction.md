# Reproduction workflow

The workflow acquires source data, extracts model states, fits readouts, and runs intervention and sensitivity analyses. Full extraction requires the source datasets, publisher-controlled model checkpoints, a Linux/CUDA environment, and substantial compute and storage. Keep generated files in the ignored `data/`, `models/`, `cache/`, `results/`, `reports/` and `manifests/private/` directories.

## 1. Check the code and acquire public inputs

Use Python 3.10 and the installation in the top-level README. Run the non-CUDA tests first. The frozen source references are in `configs/stage0.yaml`, `configs/original/main.yaml` and `manifests/public/source_revisions.json`.

```bash
python scripts/01_fetch_public_sources.py verify --target all
```

This command checks inputs already present; it does not download missing files. `download --target sro` obtains the pinned SRO checkout. Model downloads may require publisher access, as described in the frozen Stage 0 configuration. The code checks above do not rerun the scientific analysis.

## 2. Build behavioural sessions and frozen splits

```bash
python scripts/original/01_audit_tasks.py
python scripts/original/02_build_sro_sessions.py --task all
```

The second script generates `configs/original/splits.json` from the pinned SRO data and seed. Keep this participant-level file outside Git.

## 3. Extract 8B states and fit readouts

```bash
python scripts/original/03_extract_8b_states.py --model all --task all
python scripts/original/04_fit_shared_manifold.py --model minitaur_8b --layer 24
python scripts/original/04_fit_shared_manifold.py --model llama_base_8b --layer 24
python scripts/original/04_fit_shared_controls.py full
python scripts/original/04_fit_shared_controls.py task-specific
python scripts/original/04_fit_shared_controls.py haar
python scripts/original/04_fit_shared_controls.py held-out
python scripts/original/04_fit_shared_controls.py finalize
```

The original analysis also fitted other layers and sensitivity conditions. Later stages require the model-specific cached states and readouts. Use the frozen configuration for rank and layer selection.

## 4. Test native-choice effects and Jacobian access

The original analysis scripts expose these actions:

```bash
python scripts/original/09_fit_jlens.py freeze-prompts --model all
python scripts/original/09_fit_jlens.py fit --model all
python scripts/original/09_fit_jlens.py analyze --model all
python scripts/original/10_run_interventions.py run --task all
python scripts/original/10_run_interventions.py finalize
```

The scripts and `docs/results-map.md` specify their prerequisites and outputs. The response-shuffled comparison also requires the matched control workflow in `08_train_shuffled_control.py`.

## 5. Repeat the 70B behavioural readout under NF4

`extensions/nf4_extraction/run_full.py` extracts states from the NF4 models. Set its data, model and original-analysis roots with `CM_NF4_PROJECT_ROOT`, `CM_NF4_ORIGINAL_ROOT`, `CM_NF4_INPUTS_ROOT`, `CM_NF4_PILOT_ROOT`, `CM_NF4_BASE_MODEL` and `CM_NF4_ADAPTER_MODEL`. The extraction requires a verified plan, model files and prior input manifests; an empty checkout is not sufficient.

`extensions/nf4_behavioral/setup_views.py` validates the frozen extraction output and prepares isolated model views. `CM_NF4_SNAPSHOT_ROOT` can point to this checkout; the generated `configs/original/splits.json` and verified extraction input records must also be present. `stage_shared.py` reuses the frozen 8B shared-readout algorithms for both 70B model views at layer 60. Its actions are `rank-curve`, `full`, `task-specific`, `haar`, `held-out` and `finalize`. The extension does not run 70B native interventions or Jacobian tests. NF4 estimates do not replicate exact-BF16 results.

## Figure rendering

Selected aggregate analysis tables are available in [`source_data/`](../source_data/README.md). The plotting code for the current manuscript has not yet been added, so these tables do not reproduce the final figures by themselves. `scripts/original/12_make_results.py` renders an earlier figure set.

[Peng Wang: add the plotting dependencies, one command per manuscript figure, and the expected output filenames. Record the script-to-table mapping in `results-map.md`.]
