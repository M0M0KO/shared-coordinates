# Original SRO task-audit inventory

Status: **header inventory only; no Q_k decisions and no model output**

- Candidate tasks found in both releases: 16
- Complete/retest header mismatches: 1
- Tasks with no automatically recognized response candidate: 0
- Retest access in this step: CSV header only (`nrows=0`); no retest behavior rows read.
- Participant IDs or response values are not written to this report or the draft CSV.

The candidate columns are mechanical hints, not scientific decisions. Each task still
requires source-code/instruction review, Q_k assignment, parser tests, summary-statistic
reproduction, and manual inspection of 50 complete sessions before inference.

## Header differences

- `columbia_card_task_hot`: complete-only ["performance_var"]; retest-only []
