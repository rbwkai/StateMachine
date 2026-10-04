# DWS-Bench Report Evidence Audit

This document grounds the thesis narrative in the repository as it exists on 2026-09-08. It separates verified implementation and dataset facts from empirical claims that still require model evaluation or a naturalistic audit.

## Executive Findings

- The repository implements a deterministic symbolic simulator, renderer, validator, trajectory-family constructors, factor measurement, canonical step-wise gold states, counterfactual probes, and an evaluation harness.
- The currently checked-in benchmark contains 1,800 JSONL records; after the new RQ3/RQ4 scripts are
   run, the planned regenerated suite contains 2,300 records:
  - RQ1 depth: 600 records
  - RQ2 revision: 400 records
   - RQ3 distractor/interference: 400 records (including a matched `N=4` condition)
   - RQ4 entity load: 400 records
  - RQ5 structural pilot: 500 records
- The repository does not contain model result files, a `results/` directory, a ProPara dataset, or a naturalistic-audit output. Model accuracy, failure-onset, error-dynamics, transfer, and final RQ conclusions must therefore remain pending.
- The current project design has five research questions, not four. RQ4 is entity-load analysis in the README, while the implemented structural-operation study is RQ5.
- The measured metadata contains `E_actual`, `T_actual`, `D_actual`, `V_actual`, `L_word`, `L_actual`, and `N_actual`.
- The frozen base benchmark contains 1,800 records with `N_actual=0`; the checked-in RQ3 slice
   now adds 100 matched records with `N_actual=4`.
- One hundred structural-pilot records (`split_chain`) differ from their requested factors: requested `T=8,D=0`, realized `T_actual=7,D_actual=1`. This should be reported as a realized-factor discrepancy, not hidden.
- RQ1 `basic_chain` records have nonzero `V_actual` at several depths. Therefore, the implementation does not support describing the entire RQ1 dataset as `V=0` in realized terms; it supports `D_actual=0` and the requested basic-chain condition.

## Verified Dataset Summary

| Slice | Records | Family/families | Requested design |
|---|---:|---|---|
| RQ1 | 600 | `basic_chain` | `E=1`, `D=0`, `T={2,4,6,8,12,16}`; 100 each |
| RQ2 | 400 | `revision` | `E=1`, `D=0`, `T={4,8,12,16}`, `V_min=2`; 100 each |
| RQ3 | 400 planned | `interleaved_chain` | `E=3`, `T=8`, `D={4,8,16}` plus matched `D=4,N=4` |
| RQ4 | 400 planned | `interleaved_chain` | `E={2,3,4,5}`, `T=8`, `D=4`; 100 each |
| RQ5 | 500 | five structural families | `T=8` requested; 100 each |
| Total | 2,300 planned | eight families | Aggregated by `generate_all.py` |

The eight families are `basic_chain`, `revision`, `interleaved_chain`, `split_chain`, `merge_chain`, `swap_chain`, `undo_chain`, and `undo_redo_chain`.

## Realized Factor Summary

Across the currently checked-in 1,800 records:

| Factor | Minimum | Maximum | Mean | Notes |
|---|---:|---:|---:|---|
| `E_actual` | 1 | 3 | 1.50 | Unique entities instantiated across trajectory lifetime |
| `T_actual` | 2 | 16 | 8.39 | Post-initialization operations changing the target state |
| `D_actual` | 0 | 16 | 1.61 | Post-initialization operations not changing the target state |
| `V_actual` | 0 | 14 | 6.22 | Genuine target-location revisits |
| `L_word` | 32 | 312 | 130.61 | Word-count proxy, not tokenizer-specific token count |
| `N_actual` | 0 | 0 | 0.00 | No textual distractors in the frozen data |

By slice:

- RQ1: `T_actual={2,4,6,8,12,16}`, `E_actual=1`, `D_actual=0`, `L_word=32..200`.
- RQ2: `T_actual={4,8,12,16}`, `E_actual=1`, `D_actual=0`, `V_actual>=2`, `L_word=56..200`.
- RQ3: `T_actual=8`, `E_actual=3`, `D_actual={4,8,16}`, `V_actual=5`, `L_word=168..312`.
- RQ5: `T_actual={7,8}`, `E_actual={1,2}`, `D_actual={0,1}`, `V_actual={5,6}`, `L_word=90..112`.

## What the Report Can Claim Now

The report can state that:

1. The simulator and replay engine define the authoritative symbolic state.
2. Rendering occurs after symbolic trajectory construction and factor measurement.
3. The generator records requested and realized factors separately.
4. Step-wise gold states are derived by replaying the canonical operations.
5. Counterfactual probes delete a selected operation and recompute the answer through replay.
6. The eight trajectory families are structurally validated and deterministic for the tested paths.
7. The master verification suite passes the repository's smoke, invariant, factor-measurement, analysis, and evaluation-pipeline tests.
8. The checked-in benchmark has 1,800 records; the planned regenerated suite has 2,300 records after
the RQ3/RQ4 additions.
9. Generation verification is implementation evidence, not evidence that models degrade with depth or interference.

## What Must Remain Pending

Do not present the following as observed findings until the corresponding artifacts exist:

- Final-answer accuracy for any model.
- Step-wise model accuracy.
- Accuracy curves by `T`, `D`, `V`, or `E`.
- AIC-selected model curves based on real model outputs.
- Failure-onset values such as `L_T`, `L_D`, `L_V`, or `L_E`.
- Local, propagating, cancellation, or final-only error distributions from model predictions.
- Claims that textual distractors affect performance; `N_actual` is zero in the current benchmark.
- Claims about transfer to ProPara or any naturalistic corpus.
- Causal claims that factors are independent. The repository measures factors, but it does not establish perfect orthogonality.

## Required Thesis Corrections

### Research-question structure

Use five questions if the report follows the repository:

- RQ1: temporal depth.
- RQ2: state revision.
- RQ3: state-changing distractor interference.
- RQ4: entity load.
- RQ5: structural-operation pilot.

RQ4 is now retained as a primary question and has a dedicated `E={2,3,4,5}` sweep. Do not describe
RQ4 as structural diversity; the structural-operation study is RQ5.

### Terminology

- Use `L_word` or “rendered word-count proxy” for the current length field. Calling it tokenizer token length is inaccurate.
- Distinguish requested factors from realized factors in every results table.
- Describe `N_actual` as zero for the frozen benchmark; textual-distractor evaluation is not yet implemented in the current data.
- Describe the structural study as a pilot. It is not a matched-depth causal comparison because `split_chain` realizes `T_actual=7` and `D_actual=1`.
- Avoid saying that RQ1 has realized `V=0`; the actual `V_actual` values include revisits. Say that RQ1 uses the `basic_chain` family with requested `D=0` and report realized `V_actual`.

### Model list

The code registry contains these core aliases:

- `qwen2.5-0.5b` -> `Qwen/Qwen2.5-0.5B-Instruct`
- `qwen2.5-3b` -> `Qwen/Qwen2.5-3B-Instruct`
- `qwen2.5-7b` -> `Qwen/Qwen2.5-7B-Instruct`
- `llama-3.2-3b` -> `meta-llama/Llama-3.2-3B-Instruct`
- `olmo-2-1b` -> `allenai/OLMo-2-1B`

The README also mentions the 7B OLMo model, but it is registered as optional (`olmo-2-7b`) rather than as a core model. The thesis must choose one explicit five-model evaluation set and match the actual commands and outputs.

### Chapter numbering and appendices

The pasted LaTeX comments and chapter references should be reconciled. The visible structure contains more than ten numbered chapters before the audit material, while the compliance table says “Chapters 1--10.” Decide whether weekly summaries and implementation artifacts are appendices and use explicit `\appendix` where appropriate.

### LaTeX prerequisites

The report source uses `\toprule`, `\midrule`, and `\bottomrule`; `report/main.tex` now explicitly loads `booktabs`. The source is under `report/main.tex`, with `report/citations.bib` and local class/style files. Compilation has not been verified because `pdflatex` and `biber` are unavailable in the current environment; referenced image files also need to be present under the report image path.

## Recommended Results Chapter Structure

1. **Benchmark construction and verification**
   - Report the checked-in 1,800-record composition and the planned 2,300-record regenerated suite.
   - Report test-suite status and dry-run reachability.
   - Explain symbolic replay and renderer independence.

2. **Realized-factor audit**
   - Provide the factor summary table.
   - Provide requested-versus-realized mismatch counts.
   - Explicitly report `N_actual=0` for the checked-in data and `N_actual=4` for the new matched RQ3 condition.
   - Discuss `split_chain` and RQ1 revision discrepancies.

3. **Model evaluation**
   - Add one subsection per evaluated model only after prediction files exist.
   - Report exact configuration, model revision/identifier, device, precision, batch size, and seed.

4. **RQ1--RQ5 analyses**
   - Use realized factors for grouping and requested conditions for design labels.
   - Report uncertainty or counts alongside accuracy.
   - Treat RQ5 as pilot evidence, not a causal conclusion.

5. **Limitations**
   - Current absence of textual distractors.
   - Word-count rather than tokenizer-specific length.
   - Residual coupling among trajectory factors.
   - No current naturalistic audit artifact.
   - Structural-pilot mismatch for `split_chain`.

## Citation Mapping

The supplied bibliography is relevant as follows:

- Kim and Schuster (2023): entity tracking, trivial examples, and the motivation for nontrivial state changes.
- Rezaee et al. (2025): controlled state-tracking depth and model-scale comparisons.
- Tang et al. (2026): mechanistic/entity-tracking motivation; verify publication status and bibliographic metadata before final submission.
- Li et al. (2021): implicit representations and representation-level background.
- Careaga and Aksoy (2023), Chao et al. (2023), MacKenzie and Buxton (1992), and Norman (2013): these do not appear connected to the current DWS-Bench implementation or claims in the repository. Include them only if the thesis text contains a genuine discussion that requires them.

The report source now references the present bibliography file `report/citations.bib`. Bibliography validation and citation cleanup still require a LaTeX toolchain and a review of which supplied sources genuinely support the thesis text.

## Evidence Inventory

- Implementation: `world/`, `generator/`, `render/`, `pipeline.py`, `trajectory.py`.
- Experiment generation: `experiments/rq1_depth.py`, `experiments/rq2_revision.py`, `experiments/rq3_distractor.py`, `experiments/rq4_entity_load.py`, `experiments/rq5_pilot.py`, `generate_all.py`.
- Evaluation: `eval/engine.py`, `eval/eval_harness.py`, `eval/models.py`, `run_eval.py`.
- Analysis: `analysis/failure_onset.py`, `analysis/first_error.py`, `analysis/query_analysis.py`.
- Frozen data: `data/full_benchmark.jsonl` and the four slice files.
- Verification: `test/run_all.py` and the tests under `test/`.

## Next Artifacts Needed for a Complete Empirical Report

1. Prediction JSONL and metrics JSON for every evaluated model and slice.
2. The exact model registry choice, including whether OLMo-1B or OLMo-7B is evaluated.
3. A Phase 1 factor-correlation report generated from the frozen data.
4. Step-wise prediction format, if step-wise model accuracy is required.
5. The ProPara/naturalistic audit data and its mapping results.
6. All referenced report images and a LaTeX toolchain for compilation and final prose integration.
