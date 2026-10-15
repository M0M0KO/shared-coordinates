# Manuscript figures and analysis code

The table identifies the analysis behind each manuscript figure. The selected group-level outputs available in this repository are listed separately from the outputs still needed for panel-level reproduction.

| Manuscript item | Analysis source | Available numerical outputs | Plotting code to add |
| --- | --- | --- | --- |
| Figure 1 | Conceptual schematic | No model output | [Peng Wang: add the editable schematic source and export command.] |
| Figure 2: 8B shared readouts | `scripts/original/03_extract_8b_states.py`, `04_fit_shared_manifold.py`, `04_fit_shared_controls.py`; `src/cognitive_manifold/geometry/` | Selected 8B shared-readout, control and retained-information tables in [`source_data/`](../source_data/README.md) | [Peng Wang: add the Figure 2 script, chosen model/rank, panel inputs and output filenames.] |
| Figure 3: native-choice edits | `scripts/original/10_run_interventions.py`; `src/cognitive_manifold/models/residual_intervention.py` | Aggregate intervention statistics in `source_data/`. Trial-level intervention records remain outside Git. | [Peng Wang: add the Figure 3 script and a shareable aggregate table for each panel.] |
| Figure 4: downstream accessibility | `scripts/original/09_fit_jlens.py`; response-shuffled control from `08_train_shuffled_control.py` | Jacobian access table and aggregate statistics in `source_data/` | [Peng Wang: add the Figure 4 script, panel selection and null-summary inputs.] |
| Figure 5: 70B NF4 readouts | `extensions/nf4_extraction/` followed by `extensions/nf4_behavioral/stage_shared.py` | Layer-60 shared-readout, control and aggregate endpoint tables for both 70B models in `source_data/` | [Peng Wang: add the Figure 5 script, chosen ranks, panel inputs and output filenames.] |

The available tables allow numerical inspection of the analyses. They do not yet reproduce the manuscript figures directly. `scripts/original/12_make_results.py` renders an earlier figure set; the current plotting code will be added separately.
