# Generator Behavioral Audits

Auditing scripts that measure emergent generator behavior at scale. These are **not
tests** — they produce evidence for thesis claims by running thousands of
trajectories and printing summary statistics.

Last verified: 2026-10-05, commit `447116e`, with `.venv/bin/python`.

## Purpose

| Category | Scripts |
|---|---|
| **Structural causality** | `01_last_move_and_ablation.py`, `02_V_split_D_condition.py` |
| **Probe accounting** | `04_probes_redo_counterfactual.py`, `07_scoring_cases.py` |
| **Rendering & length** | `03_distractor_placement.py`, `06_render_and_length.py` |
| **Revision heuristics** | `05_revision_heuristics.py` |
| **Analysis integration** | `08_analysis_queryspec.py`, `09_analysis_sampler_firsterror_curves.py` |

## Run

From the repository root. The scripts are not installed as modules, so `PYTHONPATH`
must include the root:

```bash
PYTHONPATH=. .venv/bin/python analysis/audits/01_last_move_and_ablation.py
PYTHONPATH=. .venv/bin/python analysis/audits/09_analysis_sampler_firsterror_curves.py
```

## Why separate from `test/`?

| | `test/` (pytest) | `analysis/audits/` (scripts) |
|---|---|---|
| **Goal** | Enforce contract invariants | Measure emergent behavior |
| **Output** | PASS/FAIL | Summary tables, rates, distributions |
| **Seeds** | Few (25–500) | Many (500–2000+) |
| **CI** | Required | Not in CI |
| **Failures** | Block release | Inform thesis analysis |

## Reading audit 01

`01_last_move_and_ablation.py` prints, per family and $T$:

- `ans==lastMoveDst` — how often the answer equals the last `Move` destination,
  which would be a shortcut;
- `struct-ablation changes ans` — whether removing the required structural
  operation changes the answer;
- `ablation invalid` — how often the removal makes replay invalid.

Two readings matter when quoting it. The ablation column is `n/a` for non-structural
families, because only structural families declare required operations. And
`undo_redo_chain` reports 0.0% on `ans==lastMoveDst` because the trace is required to
end in `Redo`, not `Move` — that column is a shortcut detector, not an accuracy
measure.

## Use before freezing data

Run the full audit suite to verify:

- Answer ≠ last-Move-destination rates by family and $T$
- Structural ablation actually changes answers
- Probe accounting conserves all candidates
- Redo-validity produces both valid and invalid classes
- Length measurements match contract
- Curve fitting and failure onset behave

These scripts informed decisions D-001 through D-019 in
`documentation/DECISIONS.md`. Audit output is not a release gate and does not
substitute for `pytest`.