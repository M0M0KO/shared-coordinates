# Analysis tables

The files in `analysis_tables/` are group-level numerical outputs from the 8B analysis and the two 70B NF4 behavioural readouts. They contain task, fold, model, control and summary-statistic fields. We checked their schemas and categorical values before release; none of these files contains participant or trial identifiers. The original files were copied without changing their numerical contents.

| Analysis | Files | Rows or scope |
| --- | --- | --- |
| 8B shared readout | `8b_shared_manifold.parquet`, `8b_shared_controls.parquet`, `8b_retained_information.parquet` | 360, 8,560 and 480 rows |
| 8B interventions | `8b_causal_statistics.json` | Aggregate tests only |
| 8B Jacobian access | `8b_jlens_access.parquet`, `8b_jlens_statistics.json` | 9,009 rows and aggregate tests |
| 8B primary endpoints | `8b_primary_statistics.json` | Aggregate estimates and uncertainty |
| 70B NF4 shared readout | `70b_base_shared_manifold.parquet`, `70b_base_shared_controls.parquet`, `70b_centaur_shared_manifold.parquet`, `70b_centaur_shared_controls.parquet`, and the two `70b_*_primary_statistics.json` files | 360 shared-readout rows and 8,400 control rows per model, plus aggregate endpoint statistics |

The Parquet files retain the column names and types produced by the analysis. The JSON files retain the reported aggregate estimates, confidence intervals, test counts and p-values. Machine-specific source-path records were removed from the two 70B JSON files; their numerical results were preserved. The release does not include participant-level intervention records, retest results, extracted states, fitted weights or resampling draws.

The behavioural analysis uses the [Self-Regulation Ontology data](https://github.com/IanEisenberg/Self_Regulation_Ontology) at the revision listed in [`docs/data.md`](../docs/data.md). Those source data are credited to their original authors and remain under their own CC BY 4.0 terms. Model weights are not redistributed here.

These are analysis tables, not a final source-data export for each figure panel. The plotting code and panel-level selections for the current manuscript are still being prepared. Use [`docs/results-map.md`](../docs/results-map.md) to locate the analysis behind each figure. SHA-256 values for the copied files are recorded in [`manifest.json`](manifest.json).

[Peng Wang: once the plotting scripts are added, identify the exact input columns, filters and summary values used in each panel. Add any missing aggregate source tables here and update `manifest.json`.]
