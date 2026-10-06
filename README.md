# DWS-Bench

DWS-Bench is a deterministic benchmark for dynamic world-state reasoning in
small language models. It generates symbolic state-transition trajectories,
validates them against explicit structural contracts, renders them as natural
language, and evaluates model answers against simulator-derived ground truth.

The simulator is the source of truth: the renderer never defines or changes an
instance's gold answer.

## Project Structure

| Path | Purpose |
|---|---|
| `world/` | World state, operations, and replay logic |
| `generator/` | Query specifications, trajectory generation, sampling, validation, and the single instance gate |
| `render/` | Name vocabulary, narrative sentences, text distractors, questions |
| `eval/` | Prompts, model registry, inference engines, scoring, baselines, robustness |
| `analysis/` | Query analysis, first-error taxonomy, curve fits, failure onset, solubility |
| `analysis/audits/` | Standalone behavioural audit scripts (see its README) |
| `experiments/` | RQ1–RQ3 experiment scripts |
| `notebooks/` | Kaggle-facing per-model and figure notebooks |
| `documentation/` | Implementation reference, decision log, length controls, evidence audit |
| `test/` | pytest suites plus legacy standalone runner scripts |

Generated directories are **not** in version control. `data/` holds the generated
JSONL datasets and `results/` holds evaluation artifacts; both are gitignored and
reproduced by the commands below. There is currently no `report/`, `diagrams/`,
or checked-in dataset in the working tree.

## Requirements

Python 3.10 or newer is recommended. Install the declared dependencies from the
repository root:

```bash
python3 -m pip install -r requirements.txt
```

The optional model backends in `requirements.txt` are only needed for real
inference. Mock evaluation and the generation tests do not require a GPU.

## Benchmark Design

Each accepted instance records the realized factors:

- `E`: entity load
- `T`: target-relevant update depth
- `D`: state-changing distractor updates
- `N`: text-only narrative distractors
- `V`: revision complexity
- `U`: total canonical updates
- `L`: rendered word-count proxy

The benchmark supports eight operations: `PUT`, `MOVE`, `REMOVE`, `UNDO`,
`REDO`, `SPLIT`, `MERGE`, and `SWAP`.

### Research Questions

| RQ | Question | Core Experiment |
|---|---|---|
| **RQ1** | How does increasing target-relevant state depth affect the ability of small instruction-tuned LMs to recover the final world state, and does degradation depend on the type of state mutation rather than depth alone? | Vary `T` across 7 mutation structures (basic_chain, revision, split_chain, merge_chain, swap_chain, undo_chain, undo_redo_chain). Keep `E`, `D`, `N` controlled. |
| **RQ2** | When target-relevant transitions are held constant, how do state-changing distractors, narrative distractors, and context length differentially affect dynamic-state reasoning, and does superseded-information interference produce a distinct stale-state failure pattern? | Fixed target trajectory (interleaved_chain, E=3, T=8). Cross `D ∈ {4,8,16}` × `N ∈ {0,4,8,16}` with length-matched controls. Supersession via revision family. |
| **RQ3** | How do model scale and explicit reasoning prompting alter the accuracy and failure profile of dynamic-state reasoning, and does the effect of reasoning prompting depend on mutation structure? | 0.5B / 3B / 7B × Direct / Structured prompting on representative RQ1/RQ2 conditions. |

### Generated Data Sweeps

| Sweep | Conditions | Target records |
|---|---|---:|
| RQ1: Mutation & Depth | 7 families × T levels (`basic_chain` T∈{2,4,6,8,12,16}; `undo_redo_chain` T∈{6,8,12,16}; others T∈{4,8,12,16}) | 1,500 |
| RQ2: Interference & Supersession | `interleaved_chain` E=3 T=8, 4 containers; D∈{4,8,16} at N=0, N∈{4,8,16} at D=4, plus `revision` T=8 supersession | 350 |
| **Full benchmark** | Aggregated by `generate_all.py` | **1,850** |

RQ1 counts are 30 conditions × 50 instances. RQ2 is 7 conditions × 50 instances.
Note a known bookkeeping defect: `generate_all.py` declares an expected RQ2 count of
700 in `EXPERIMENT_SCRIPTS`, which does not match the 350 the RQ2 script plans. The
declared count is never read, so generation is unaffected, but the two numbers should
be reconciled before release.

These are *targets*, not achieved counts. Two families do not currently survive the
generation gate, verified 2026-10-05 by calling `experiments/_common.generate_instance`
directly:

- `merge_chain` is rejected on every attempt (`verify_factors:T`), so RQ1 would emit
  zero records for that family;
- `split_chain` is accepted at `D=0` and rejected on every attempt at `D>=1`.

`python3 generate_all.py --dry-run` passes both scripts because the reachability probe
calls `build_trajectory` and never enters the gate. See `documentation/IMPLEMENTATION.md`
§12 and §17.2.

## Generate Data

Run reachability probes without writing the benchmark:

```bash
python3 generate_all.py --dry-run
```

Both RQ1 and RQ2 currently pass their reachability probes.

Generate the complete default suite and aggregate it into
`data/full_benchmark.jsonl`:

```bash
python3 generate_all.py
```

The generated files are:

```text
data/rq1_mutation_depth/rq1_mutation_depth.jsonl   1,500 records
data/rq2_interference/rq2_interference.jsonl         350 records
data/full_benchmark.jsonl                         1,850 records
```

Every record is produced through `generator.instance.build_validated_instance`,
which runs the release gate in order: `build_trajectory` → `render_narrative` →
`measure_factors` → structural causality → `verify_factors` → `verify_length` →
answer leakage → duplicate trace → record assembly. Requested factors are never
trusted; they are measured from the canonical replay on every instance.

## Run Evaluation

The evaluation CLI supports these registered model keys:

```text
qwen2.5-0.5b    Qwen/Qwen2.5-0.5B-Instruct
qwen2.5-3b      Qwen/Qwen2.5-3B-Instruct
qwen2.5-7b      Qwen/Qwen2.5-7B-Instruct
llama-3.2-3b    meta-llama/Llama-3.2-3B-Instruct
olmo-2-1b-instruct allenai/OLMo-2-0425-1B-Instruct
phi-4-mini      microsoft/Phi-4-mini-instruct   (optional)
olmo-2-7b       allenai/OLMo-2-1124-7B-Instruct (optional)
```

Every entry in `eval/models.py` pins an immutable commit-hash revision. No model
has actually been evaluated in this working tree; there are no `results/`
artifacts.

Decoding is deterministic by contract: `temperature=0.0`, `top_p=1.0`,
`do_sample=False`, `max_new_tokens=256`. `HuggingFaceEngine.generate_batch`
enforces greedy decoding unless `enforce_greedy=False` is passed explicitly.

Run a dependency-free mock evaluation with:

```bash
python3 run_eval.py --model qwen2.5-0.5b --dataset full --mock
```

Run real inference on a supported device with:

```bash
python3 run_eval.py \
  --model qwen2.5-3b \
  --dataset full \
  --device cuda \
  --precision bfloat16
```

Dataset shortcuts are `full`, `rq1`, and `rq2`. The help text also advertises
`rq3`, but no `rq3` entry exists in the dataset map — `run_eval.py:66` maps only
`rq1`, `rq2`, and `full`. RQ3 is an evaluation-matrix experiment, not a dataset
slice.

Use `--cot` to enable the structured chain-of-thought prompt. Evaluation output
is written under `results/<model_name>/` and includes predictions JSONL, metrics
JSON, a Markdown report, and an audit CSV.

### RQ3 Scale & Reasoning Evaluation

RQ3 is evaluation-only (consumes RQ1/RQ2 data):

```bash
python3 experiments/rq3_scale_reasoning.py --all-models --all-prompts
python3 experiments/rq3_scale_reasoning.py --analyze
```

### Standalone Evaluation Utilities

Three modules are implemented and tested but **not** wired into `run_eval.py`.
They must be called from a script or notebook to take effect:

| Module | Purpose |
|---|---|
| `eval/baselines.py` | Stateless and most-frequent-class solvers, answer-distribution entropy |
| `eval/robustness.py` | Prompt-variant and paraphrase sensitivity, consistency rate |
| `analysis/solubility.py` | Rule-based surface-solubility audit (unique-answer, LLM-judged, full audit) |

`analysis/evaluate_existing_predictions.py` re-scores prediction files offline
without model inference.

## Tests

The suite is pytest-based. From the repository root:

```bash
.venv/bin/python -m pytest -q
```

Current status: 611 collected, 529 passed, 79 failed, 3 skipped.

The suite is deliberately not green. 78 of the 79 failures are tests named
`test_failsnow_*`. They encode properties that the current generator does **not**
yet satisfy — count-query gold containers, query-type serialization, length
matching, dataset manifests, baseline solver bounds, shortcut audits, paraphrase
rules, and others. They are red on purpose so the gaps stay visible. The one
unmarked failure is `test_known_findings.py::test_f10_revision_count_is_enforced`
(`TrajectorySpec.revision_count` is still not checked against measured `V`).

A legacy standalone runner also exists:

```bash
python3 test/run_all.py
```

It executes seven scripts directly (`smoke_test`, `smoke_test_trajectories`,
`test_invariants`, `test_measured_factors`, `test_analysis_and_eval`,
`test_eval_pipeline`, `test_scoring`). Two of them currently fail for code
reasons, not missing data:

- `smoke_test_trajectories.py` builds a `split_chain` spec without
  `query_type="count"`, which `build_split_chain` now requires;
- `test_invariants.py` hits an `UnboundLocalError` on `counterfactual` in
  `generator/structural.py:165-196`, a duplicated necessity block left outside the
  loop that binds it.

Both are script-level drift from the current contracts. Use pytest as the source
of truth.

## Reproducibility Notes

Generation records requested and measured factors, seeds, trace hashes, and
validation outcomes. Evaluation records model configuration and revision, prompt
version, decoding settings, extracted answers, correctness, and step-wise results.

Datasets and run outputs are deliberately not committed. Archive them alongside
the code revision and seed policy used to produce them when reporting new
experiments. No release manifest builder exists yet, so the provenance in
`requirements.md` §13 is currently assembled by hand.