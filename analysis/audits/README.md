# Generator Behavioral Audits

Auditing scripts that measure emergent generator behavior at scale. These are **not tests** — they produce evidence for thesis claims by running thousands of trajectories and printing summary statistics.

## Purpose

| Category | Scripts |
|----------|---------|
| **Structural causality** | `01_last_move_and_ablation.py`, `02_V_split_D_condition.py` |
| **Probe accounting** | `04_probes_redo_counterfactual.py`, `07_scoring_cases.py` |
| **Rendering & length** | `03_distractor_placement.py`, `06_render_and_length.py` |
| **Revision heuristics** | `05_revision_heuristics.py` |
| **Analysis integration** | `08_analysis_queryspec.py`, `09_analysis_sampler_firsterror_curves.py` |

## Run

From repo root (where `generator/`, `world/`, `render/`, `eval/`, `analysis/` live):

```bash
PYTHONPATH=. python3 analysis/audits/01_last_move_and_ablation.py
PYTHONPATH=. python3 analysis/audits/09_analysis_sampler_firsterror_curves.py
```

## Why separate from `test/`?

| | `test/` (pytest) | `analysis/audits/` (scripts) |
|---|---|---|
| **Goal** | Enforce contract invariants | Measure emergent behavior |
| **Output** | PASS/FAIL | Summary tables, rates, distributions |
| **Seeds** | Few (25–500) | Many (500–2000+) |
| **CI** | Required | Not in CI |
| **Failures** | Block release | Inform thesis analysis |

## Use before freezing data

Run the full audit suite to verify:
- Answer ≠ last-Move-destination rates by family/T
- Structural ablation actually changes answers
- Probe accounting conserves all candidates
- Redo-validity produces both valid/invalid classes
- Length measurements match contract
- Curve fitting and failure onset behave

These scripts informed decisions D-001 through D-017 in `documentation/DECISIONS.md`.