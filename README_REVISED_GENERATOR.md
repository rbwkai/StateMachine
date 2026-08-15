
# DWS-Bench revised generator

This revision changes the generation philosophy from:

    random valid trajectory -> heuristic query

to:

    sample valid trajectory -> analyze trajectory -> accept a query only if
    it satisfies an explicit QuerySpec.

## Main additions

- `analysis.py`
  - `QuerySpec`
  - `QueryAnalysis`
  - structural relevance
  - state-change count
  - initial/final answer comparison
  - interleaving score
  - valid deletion-based counterfactual sensitivity

- `generator.py`
  - candidate queries
  - QuerySpec-driven selection
  - existing counterfactual probes retained

- `pipeline.py`
  - rejection sampling over trajectories
  - analysis metadata stored in every instance

## Important methodological choice

No LLM is used to paraphrase benchmark instances. All surface text is generated
by deterministic renderers from the canonical operation trace.

The canonical simulator remains the only source of truth.

## Suggested benchmark configurations

### A. Basic sequential state tracking

```python
QuerySpec(
    query_type="location",
    must_change_from_initial=True,
    min_relevant_steps=4,
    min_state_changes=3,
    min_dependency_depth=4,
)
```

Use only `Put`, `Move`, `Remove`.

### B. Interleaving

```python
QuerySpec(
    query_type="location",
    must_change_from_initial=True,
    min_relevant_steps=5,
    min_state_changes=3,
    min_interleaving=0.4,
)
```

Use unrelated entity/container operations as structural interference.

### C. Revision

```python
QuerySpec(
    query_type="location",
    must_change_from_initial=True,
    min_relevant_steps=5,
    min_state_changes=3,
    require_revision=True,
)
```

Prefer explicit trajectory templates for revision patterns rather than relying
on random chance.

### D. History

Use `force_redo_probe=True` for the dedicated redo invalidation condition.

## What is intentionally not solved yet

`Split` creates identity/coreference ambiguity. `Merge` creates broad
multi-entity propagation. They should be treated as advanced extensions rather
than mixed into the basic state-tracking condition.

Deletion-based counterfactual sensitivity is conservative: if deleting an
operation makes a later operation invalid, that counterfactual is excluded
rather than repaired. This avoids silently changing the trajectory.

## One recommended next step

Add explicit trajectory constructors for each difficulty family
(`basic_chain`, `interleaved_chain`, `cancellation`, `revision`,
`undo_redo`) instead of expecting random sampling to produce rare patterns.
