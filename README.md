# Shared representations across cognitive tasks influence language model decisions

This repository contains analysis code for task-shared predictive directions in cognitive language models. It fits task-conditioned choice readouts to frozen model states, compares the shared subspace with full, task-specific and random-subspace controls, and tests its downstream role with finite activation edits and Jacobian-based sensitivity. An extension repeats the behavioural readout analysis with NF4-quantized 70B models.

The repository contains code, analysis configurations and selected group-level results. It does not contain the manuscript, participant-level data, model weights, extracted hidden states or server run logs.

## Documentation

| Question | Location |
| --- | --- |
| What data and models are needed? | [`docs/data.md`](docs/data.md) |
| How are the analysis stages run? | [`docs/reproduction.md`](docs/reproduction.md) |
| Which code and numerical outputs underlie each figure? | [`docs/results-map.md`](docs/results-map.md) and [`source_data/`](source_data/README.md) |
| Where did the code come from? | [`docs/provenance.md`](docs/provenance.md) |

Figure 1 is a conceptual schematic. Figures 2–4 use the 8B analysis code in `src/` and `scripts/original/`, and Figure 5 uses the 70B NF4 code in `extensions/`. See `docs/results-map.md` for the code behind each manuscript figure. The earlier `scripts/original/12_make_results.py` produces a different figure set.

[Pending: Peng Wang's plotting code for the current manuscript figures. Add the scripts, their input tables and the commands to regenerate each panel before describing this repository as a complete figure reproduction package.]

## Repository layout

```text
configs/                  Frozen analysis settings and task audit
src/cognitive_manifold/  Data parsing, state extraction, geometry, readouts and inference
scripts/original/         Frozen 8B analysis stages and original-profile utilities
extensions/nf4_extraction/  70B NF4 state-extraction source
extensions/nf4_behavioral/  70B NF4 shared-readout adapter and tests
tests/                    Unit and contract tests
manifests/public/         Public source revisions and model file inventory
docs/                     Data, reproduction, figure mapping and provenance
source_data/              Selected group-level analysis tables
```

## Installation and code checks

The frozen 8B environment targets Python 3.10 on Linux. Full model runs require NVIDIA GPU resources and separately acquired model weights. A code-level check does not download data or load a model:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python scripts/check_release.py
python -m pytest -q extensions/nf4_behavioral/test_stage_shared.py extensions/nf4_behavioral/test_setup_views.py tests/test_original_falsification.py tests/test_original_inference.py tests/test_original_shared_controls_cache.py tests/test_shared_speedups.py tests/test_redaction.py
```

The analysis uses separate, Git-ignored data, model and output directories under the repository root. The broader test suite requires the pinned Linux environment and, for some integration tests, the public SRO checkout. See [`docs/reproduction.md`](docs/reproduction.md) before a full run. The NF4 extension also requires a compatible quantization runtime and verified 70B state inputs.

## Data and model access

The behavioural inputs come from the public [Self-Regulation Ontology repository](https://github.com/IanEisenberg/Self_Regulation_Ontology) at the revision recorded in `configs/original/main.yaml`. The [Psych-101 resource](https://huggingface.co/datasets/marcelbinz/Psych-101) supplies task materials. Obtain model weights from their publishers under the applicable licences; this repository does not redistribute datasets or model checkpoints. `scripts/original/02_build_sro_sessions.py` generates the development/confirmation split from the public SRO input. Participant-level split records are not committed.

## Scope of the 70B analysis

The exact-BF16 70B analysis could not be completed with the available resources. The 70B results use NF4-quantized models and do not replicate an exact-BF16 analysis. They cover behavioural readouts, not 70B native interventions or Jacobian analyses.

## License

Original material in this repository is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), unless otherwise noted. See [`LICENSE`](LICENSE). Third-party datasets, model weights and dependencies retain their own terms and are not covered by this license.
