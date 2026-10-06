# DWS-Bench Report Evidence Audit

**Repository state audited:** 2026-10-05, commit `447116e`
**Purpose:** state what the repository can evidence now, and what must remain
pending until the corresponding artifacts exist.

This document replaces the 2026-09-08 audit, which described a five-RQ design and a
checked-in 1,800-record benchmark. Both are obsolete: the design is now three RQs
(`documentation/DECISIONS.md` D-018), and no dataset is in version control
(`data/` and `results/` are gitignored and absent from the working tree).

---

## Executive findings

- The repository implements a deterministic symbolic simulator, a canonical replay
  engine, an eight-family trajectory generator, a factor measurer, a single
  validated-instance gate, a narrative renderer, an evaluation harness with a model
  registry, and an analysis layer for curves, failure onset, and first-error
  taxonomy.
- **There is no dataset and there are no model results.** No accuracy number,
  curve fit, failure-onset value, or error-taxonomy distribution can be claimed.
- The current design has three research questions, not five. See D-018 for the
  RQ4/RQ5 merge.
- The planned dataset is 1,850 records: 1,500 from RQ1 and 350 from RQ2. This is a
  *plan* verified by dry-run reachability, not a released dataset.
- The pytest suite is deliberately red: 611 collected, 529 passed, 79 failed, 3
  skipped. 78 failures are `test_failsnow_*` markers encoding known gaps; the 79th
  is the unenforced `revision_count` (F10).
- `SPEC.md` declares version `0.3.0` while every generated record is stamped
  `0.2.0-v2`. Provenance is therefore internally inconsistent until reconciled.
- Family minimum `T` values in `SPEC.md` disagree with
  `generator/trajectory_validation.py` for three families. The contract wins, but
  the code must be corrected before a freeze.
- `generator/structural.py` contains a live defect (duplicated necessity block,
  `UnboundLocalError` when every deletion of a required op is invalid). Measured:
  `split_chain` is accepted at `D=0` and rejected on every attempt at `D>=1`.
- Count-query gold is implemented only for `split_chain`. `merge_chain`,
  `swap_chain`, and `undo_chain` accept `query_type="count"` and emit location
  questions instead, and the record does not carry `query_type`, so nothing
  downstream can detect the substitution.
- `merge_chain` never passes the gate: `verify_factors:T` fails on every attempt at
  every `$T$` tested. RQ1 would emit zero `merge_chain` records, and the dry-run
  reachability probe cannot see this because it never enters the gate.

---

## Verified implementation facts

| Fact | Evidence |
|---|---|
| 8 operations: `Put`, `Move`, `Remove`, `Split`, `Merge`, `Swap`, `Undo`, `Redo` | `world/operations.py` |
| `apply_op` is the single mutation path | `world/operations.py` |
| One `replay_trace` pass feeds gold, step-wise gold, and rendering | `world/replay.py` |
| 8 trajectory families in 5 capability groups | `generator/dataset_spec.py` |
| Every record goes through `build_validated_instance` | `generator/instance.py` |
| Gate order: build, render, measure, structural, factors, length, leakage, dedup | `generator/instance.py:77-84` |
| Requested factors are measured, never trusted | `generator/metadata.py` |
| RQ1 and RQ2 both pass the reachability probe | `generate_all.py --dry-run` |
| 7 models registered, each pinned to a commit hash | `eval/models.py` |
| Deterministic decoding enforced at the engine | `eval/engine.py` |
| One scoring authority | `eval/scoring.py` |
| $\tau = 0.70$, $L_{max} = 600$ words, $\tau$ in one constants module | `generator/constants.py` |
| One renderer surface; `render/templates.py` deleted | `render/narrative.py` |
| Datasets and results are not versioned | `.gitignore` |

---

## Planned dataset (not released)

| Sweep | Conditions | Target records | Source |
|---|---:|---:|---|
| RQ1 mutation & depth | 7 families × `T` levels | 1,500 | `experiments/rq1_mutation_depth.py` |
| RQ2 interference & supersession | 3 `D` levels at `N=0`; 3 `N` levels at `D=4`; 1 supersession | 350 | `experiments/rq2_interference.py` |
| Full benchmark | aggregated | 1,850 | `generate_all.py` |

RQ1 detail, 50 instances per condition:

| Family | E | D | N | T levels | Query |
|---|---:|---:|---:|---|---|
| `basic_chain` | 1 | 0 | 0 | 2, 4, 6, 8, 12, 16 | location |
| `revision` | 1 | 0 | 0 | 4, 8, 12, 16 | location |
| `split_chain` | 2 | 0 | 0 | 4, 8, 12, 16 | count |
| `merge_chain` | 2 | 0 | 0 | 4, 8, 12, 16 | count |
| `swap_chain` | 2 | 0 | 0 | 4, 8, 12, 16 | count |
| `undo_chain` | 1 | 0 | 0 | 4, 8, 12, 16 | count |
| `undo_redo_chain` | 1 | 0 | 0 | 6, 8, 12, 16 | location |

RQ2 detail: `interleaved_chain`, E=3, T=8, 4 containers, `N` levels {4, 8, 16} at
D=4, plus a `revision` T=8 supersession condition.

Bookkeeping defect: `generate_all.py` records an expected RQ2 count of 700, which
does not match the 350 conditions the RQ2 script actually plans. It affects
reporting only, not generation.

---

## What the thesis may claim now

1. The simulator and canonical replay define the authoritative symbolic state, and
   rendering never re-derives state.
2. Every generated instance passes an ordered gate that measures its factors from
   the replay and rejects leakage, duplicates, and length violations.
3. Requested and measured factors are recorded separately on every record.
4. Step-wise gold is derived from the same replay trace used for narration.
5. Eight families are structurally validated, with required operations proven
   causally necessary for the queried answer.
6. Generation is deterministic: same seed and same revision produce byte-identical
   records (`test/test_experiment_scripts.py` covers the rerun case).
7. The planned design is reachable: 30 RQ1 conditions and 7 RQ2 conditions pass the
   reachability probe.

## What must remain pending

- Final-answer or step-wise accuracy for any model.
- Accuracy curves in `T`, `D`, `V`, `E`, or `N`.
- AIC-selected curve fits based on real model output.
- Failure-onset values $L_f$ for any factor.
- First-error taxonomy distributions.
- Any claim that text distractors affect performance. RQ1 has `N=0` throughout, so
  the `N` axis exists only in RQ2.
- Any claim that length matching worked. `experiments/rq2_interference.py` sets
  `LENGTH_MATCHED = True`, but the corresponding test
  (`test_failsnow_rq2_cells_are_length_matched_within_three_words`) fails, and the
  `matched_*` record fields described in `documentation/LENGTH_CONTROLS.md` are not
  emitted.
- Any claim that no solver beats chance. The shortcut audits are mostly red
  (`test_shortcut_audits.py`, 19 `test_failsnow_*`).
- Any claim that factor effects are independent. Factors are measured, not
  orthogonalised.
- Transfer to any naturalistic corpus. No such artifact exists.

---

## Terminology corrections for the thesis

- Use `L_word` or "rendered word-count proxy". `L_actual` was removed (D-006 in the
  decision log) precisely because it duplicated `L_word` under a second name.
- Distinguish requested from measured factors in every table.
- Describe the three-RQ structure, and cite D-018 if the five-RQ framing appears in
  earlier drafts.
- State the test baseline honestly, including the `test_failsnow_*` convention. A
  red suite here is a documented gap register, not a hidden failure.
- Do not describe the structural gate as clean: `generator/structural.py` has an
  open defect on `split_chain`.

## Model list

Core, per `eval/models.py`:

| Alias | HF id | Revision pinned |
|---|---|---|
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | yes |
| `qwen2.5-3b` | `Qwen/Qwen2.5-3B-Instruct` | yes |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | yes |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | yes |
| `olmo-2-1b-instruct` | `allenai/OLMo-2-0425-1B-Instruct` | yes |

Optional: `phi-4-mini` (`microsoft/Phi-4-mini-instruct`), `olmo-2-7b`
(`allenai/OLMo-2-1124-7B-Instruct`). Every entry pins a commit hash; no floating
`main`.

Choose one explicit evaluation set and match the recorded commands. No model has
been run in this working tree.

---

## Evidence inventory

- Simulator: `world/state.py`, `world/operations.py`, `world/replay.py`
- Generator: `generator/trajectories.py`, `generator/trajectory_validation.py`,
  `generator/metadata.py`, `generator/structural.py`, `generator/instance.py`
- Renderer: `render/names.py`, `render/narrative.py`
- Probes: `generator/probes.py`
- Experiment layer: `experiments/_common.py`, `experiments/rq1_mutation_depth.py`,
  `experiments/rq2_interference.py`, `experiments/rq3_scale_reasoning.py`,
  `generate_all.py`
- Evaluation: `eval/models.py`, `eval/engine.py`, `eval/scoring.py`,
  `eval/eval_harness.py`, `eval/baselines.py`, `eval/robustness.py`,
  `run_eval.py`
- Analysis: `analysis/query_analysis.py`, `analysis/first_error.py`,
  `analysis/failure_onset.py`, `analysis/curves.py`, `analysis/solubility.py`,
  `analysis/evaluate_existing_predictions.py`
- Audits: `analysis/audits/01`–`09` with its own README
- Verification: `test/` under pytest, plus the legacy `test/run_all.py`
- Notebooks: `notebooks/run_all_models.ipynb` and per-model prompt variants,
  `notebooks/generate_thesis_figures.ipynb`. These expect `data/full_benchmark.jsonl`
  and a `report/` tree that do not exist in the working tree.
- `paper/` is gitignored and is not part of the implementation surface.

## Next artifacts needed for a complete empirical report

1. A release manifest with commit, versions, seed policy, condition list, dependency
   lock, accepted/rejected counts, and dedup statistics.
2. Resolution of `SPEC.md` OPEN-15 through OPEN-18.
3. A green run of the non-`failsnow` suite plus an explicit decision on each
   `failsnow` marker.
4. A generated dataset, then predictions for one explicit model set.
5. RQ2 length-matching actually implemented, with `matched_*` metadata emitted.
6. Baselines, robustness, and solubility wired into `run_eval.py` or invoked
   explicitly with recorded output.
7. A factor-correlation report over the realized dataset, using the rule in
   `documentation/LENGTH_CONTROLS.md`.