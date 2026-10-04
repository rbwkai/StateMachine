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
| `generator/` | Query specifications, trajectory generation, sampling, and validation |
| `render/` | Natural-language rendering and templates |
| `eval/` | Model configurations, inference engines, and evaluation harness |
| `analysis/` | Accuracy analysis, failure onset, and error classification |
| `experiments/` | RQ1–R3 dataset-generation scripts |
| `data/` | Generated JSONL benchmark datasets |
| `results/` | Evaluation predictions, metrics, and reports |
| `diagrams/` | Conceptual thesis-diagram source files |
| `test/` | Smoke, invariant, factor, analysis, and pipeline tests |

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

| Sweep | Conditions | Est. Records |
|---|---|---:|
| RQ1: Mutation & Depth | 7 families × T∈{2,4,6,8,12,16} (basic_chain); T∈{4,8,12,16} (others) | ~1,500 |
| RQ2: Interference & Supersession | D×N crossed + supersession (revision T=8) | ~700 |
| **Full benchmark** | Aggregated | **~2,200** |

## Generate Data

Run reachability probes without writing the benchmark:

```bash
python3 generate_all.py --dry-run
```

Generate the complete default suite and aggregate it into
`data/full_benchmark.jsonl`:

```bash
python3 generate_all.py
```

The generated files are:

```text
data/rq1_mutation_depth/rq1_mutation_depth.jsonl   ~1,500 records
data/rq2_interference/rq2_interference.jsonl       ~700 records
data/full_benchmark.jsonl                         ~2,200 records
```

## Run Evaluation

The evaluation CLI supports these registered model keys:

```text
qwen2.5-0.5b
qwen2.5-3b
qwen2.5-7b
llama-3.2-3b (not evaluated)
olmo-2-1b (not evaluated)
```

The default generation configuration uses deterministic decoding and
`max_new_tokens=256`. Run a dependency-free mock evaluation with:

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

Dataset shortcuts currently include `full`, `rq1`, `rq2`.

Use `--cot` to enable the structured chain-of-thought prompt. Evaluation output
is written under `results/<model_name>/` and includes predictions JSONL, metrics
JSON, a Markdown report, and an audit CSV.

### RQ3 Scale & Reasoning Evaluation

RQ3 is evaluation-only (consumes RQ1/RQ2 data):

```bash
python3 experiments/rq3_scale_reasoning.py --all-models --all-prompts
python3 experiments/rq3_scale_reasoning.py --analyze
```

## Tests

Run the master test and smoke-test collection:

```bash
python3 test/run_all.py
```

The master runner covers the smoke, trajectory, invariant, measured-factor,
analysis, and evaluation-pipeline tests.

## Thesis Figures and Report

The six conceptual diagrams in `diagrams/` use the local renderer in
`diagrams/helpers.py`. Generate them from the repository root:

```bash
for script in diagrams/fig*.py; do
  python3 "$script" || exit 1
done
```

The scripts write PNG and PDF files to `report/images/thesis_figures/`. That
directory is no longer part of this repository, so point the scripts elsewhere or
delete them if the diagrams are not needed.

## Reproducibility Notes

Generation records requested and realized factors, random seeds where applicable,
and validation outcomes. Evaluation records model configuration, prompt mode,
decoding settings, extracted answers, final correctness, and step-wise results.
Keep generated data and result directories versioned or archived with the code
revision used to produce them when reporting new experiments.