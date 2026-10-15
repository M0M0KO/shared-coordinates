# Code provenance

The 8B analysis code comes from `cognitive-manifold` at commit `32cb897`. The 70B NF4 extractor and behavioural readout code were developed later using a checked copy of the same analysis code. The shared algorithms in this repository were checked against that source before preparation for release.

The release changes are limited to paths and one test prerequisite:

- `configs/original/main.yaml` no longer contains two paths to private source documents. Its numerical settings are unchanged.
- The NF4 extractor accepts paths to the data and model files through environment variables. Its extraction method, selected layers and quantization settings are unchanged.
- The 70B behavioural readout locates the original code through `CM_NF4_SNAPSHOT_ROOT`. Its checks on the input files are unchanged.
- The raw-SRO integration test runs when the SRO data have been downloaded. It is skipped otherwise.

The repository excludes model weights, hidden states, participant-level records, machine-specific run state and logs. It includes selected group-level analysis tables in [`source_data/`](../source_data/README.md). See [`reproduction.md`](reproduction.md) for the inputs and environment needed to rerun the analysis.
