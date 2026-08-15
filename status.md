Based on the smoke-test output you provided, the current implementation is internally consistent across the major execution paths.

The important result is that this is not just an import test. You tested the actual state machine, replay semantics, query-selection constraints, trajectory gold generation, counterfactual intervention, redo history, pipeline integration, and deterministic generation.

Current status:

| Component                  | Status | What was verified                                                                     |
| -------------------------- | ------ | ------------------------------------------------------------------------------------- |
| `world/` state model       | PASS   | `Put → Move → Undo → Redo`                                                            |
| Undo/redo semantics        | PASS   | Redo stack invalidated after a new action                                             |
| Replay engine              | PASS   | Operations replay to the expected state                                               |
| Invalid-operation handling | PASS   | Invalid `Move`/`Undo` rejected                                                        |
| `generator/sampler.py`     | PASS   | Valid sequences generated under constraints                                           |
| Query selection            | PASS   | `QuerySpec` actually filters candidate queries                                        |
| `analysis.py`              | PASS   | Relevant steps, dependency depth, interleaving, revisions, counterfactual sensitivity |
| `trajectory.py`            | PASS   | Per-step state/query gold agrees with final state                                     |
| Counterfactual probes      | PASS   | Original vs. intervention answers correctly compared                                  |
| `pipeline.py`              | PASS   | Complete benchmark record generated                                                   |
| Redo-validity              | PASS   | Dedicated history condition works                                                     |
| Determinism                | PASS   | Same seed produces identical record                                                   |

One particularly useful output is:

```text
relevant_steps: [4, 5, 6]
state_change_count: 3
dependency_depth: 7
interleaving_score: 0.2857
counterfactual_sensitive_steps: [6]
```

This demonstrates that your `QuerySpec` machinery is doing something substantive. It is not merely selecting an arbitrary final object. The selected query has a measurable trajectory structure and a causally sensitive operation.

The earlier `KeyError('gold_answer')` problem also appears to be fixed. Your current counterfactual records now have the intended schema:

```text
{
    "remove_step": 4,
    "removed_sentence": "...",
    "original_answer": "c1",
    "counterfactual_answer": "c0",
    "answer_changed": true
}
```

and the invariant

```python
answer_changed == (
    original_answer != counterfactual_answer
)
```

passes.

There is, however, an important distinction:

**The smoke tests establish internal consistency, not yet benchmark validity.**

You have now established:

```text
Simulator
    ↓
Replay
    ↓
Analysis
    ↓
Query selection
    ↓
Trajectory gold
    ↓
Natural-language rendering
    ↓
Distractors
    ↓
Counterfactual probes
    ↓
Final benchmark record
```

is internally coherent.

The next verification stage should therefore be adversarial/property testing rather than more basic smoke tests.

I would do these next.

### 1. Test every operation type

Your smoke test currently demonstrates several operations, but you should explicitly exercise:

```text
Put
Move
Remove
Split
Merge
Swap
Undo
Redo
```

especially combinations involving:

```text
Split → Move
Merge → Move
Swap → Move
Remove → Undo
Move → Undo → Redo
Move → Undo → new Move → Redo
```

This matters because `Split`, `Merge`, and `Swap` have substantially different state-transition semantics.

### 2. Test all QuerySpec dimensions

You should generate batches for:

```python
QuerySpec(
    query_type="location",
    must_change_from_initial=True,
    min_relevant_steps=1,
)
```

then progressively:

```python
min_relevant_steps=3
min_relevant_steps=5
min_state_changes=3
min_dependency_depth=4
min_interleaving=0.4
require_revision=True
min_undo=1
min_redo=1
```

and verify that every accepted record actually satisfies the requested constraint.

This is especially important because `select_query()` itself is correct only if `analysis.matches()` correctly implements every field.

### 3. Test rejection behavior

For intentionally impossible specifications:

```python
QuerySpec(
    query_type="location",
    min_relevant_steps=100,
)
```

the generator should eventually fail with a clear:

```text
RuntimeError:
could not generate a trajectory satisfying QuerySpec(...)
```

rather than returning an invalid example.

This verifies the constraint-first philosophy.

### 4. Test counterfactual validity

You already handle this correctly in principle:

```python
counterfactual_gold(...) -> None
```

when removing an operation causes a later operation to become invalid.

You should explicitly test cases such as:

```text
Put A
Move A
Move A
```

and remove the initial `Put`.

The resulting replay should be rejected, and that intervention should **not** appear as a valid counterfactual probe.

This is an important methodological property because you don't want your counterfactual operation to silently alter the entire semantics of the trajectory.

### 5. Test distractor invariance

This is particularly important for your DWS-Bench design.

For the same:

```text
seed
operations
query
```

generate:

```text
N = 0
N = 5
N = 20
N = 50
```

and verify:

```python
gold_answer_N0 == gold_answer_N5 == ...
trajectory_N0 == trajectory_N5 == ...
```

while only the surface `sentences` change.

That experimentally confirms that your `N` factor is genuinely a **textual interference factor**, rather than accidentally modifying the symbolic world.

### 6. Test determinism more aggressively

You currently have:

```text
Same seed -> identical record: PASS
```

Good.

But test:

```text
seed 1 → record A
seed 1 → record B
```

across maybe 100–1000 examples.

Then check complete JSON equality.

Also test that different seeds actually produce diversity. You don't want accidental determinism caused by a hidden fixed ordering.

### 7. Run a batch-generation invariant test

This is probably the most valuable test before moving on to model evaluation.

Generate, for example:

```text
1,000 examples
```

and assert for every record:

```python
len(operations) == U
len(trajectory) == U
len(step_wise_gold) == U

gold_answer == step_wise_gold[-1]

trajectory[-1]["state_after"] == final_state

analysis.relevant_steps satisfy QuerySpec

counterfactual.answer_changed ==
    original_answer != counterfactual_answer
```

Then collect:

```text
generation failure rate
average attempts
query-type distribution
operation-type distribution
relevant-step distribution
dependency-depth distribution
interleaving distribution
counterfactual-sensitive rate
```

That will tell you whether the generator is merely correct or also **usable as a benchmark generator**.

### One thing I would flag in the current architecture

Your project currently has both:

```text
generator/sampler.py
```

and:

```text
sampler.py
```

at the repository root.

Likewise, you have the rendering code consolidated under:

```text
render/
    __init__.py
    names.py
    narrative.py
    templates.py
```

but your pasted history shows multiple versions of rendering functions.

Before doing large-scale generation, I would clean up any duplicate/legacy modules so there is exactly one authoritative implementation for each responsibility.

Your intended architecture should be:

```text
StateMachine/
│
├── world/
│   ├── __init__.py
│   ├── errors.py
│   ├── operations.py
│   ├── replay.py
│   └── state.py
│
├── generator/
│   ├── __init__.py
│   ├── probes.py
│   └── sampler.py
│
├── render/
│   ├── __init__.py
│   ├── names.py
│   ├── narrative.py
│   └── templates.py
│
├── analysis.py
├── generator.py
├── pipeline.py
├── trajectory.py
├── example.py
│
└── tests/
    ├── test_world.py
    ├── test_sampler.py
    ├── test_queries.py
    ├── test_trajectory.py
    ├── test_counterfactual.py
    ├── test_pipeline.py
    └── test_invariants.py
```

So, at this point I would **not change the core logic just because the smoke tests pass**. The next step is to stress-test the implementation and verify the statistical properties of the generated benchmark.

The current result can reasonably be summarized as:

> **DWS-Bench generator v1 is functionally passing its smoke-test suite. The simulator, replay engine, constraint-based query selection, trajectory gold generation, counterfactual probes, redo-validity condition, pipeline integration, and deterministic behavior are mutually consistent on the tested paths. The remaining verification task is adversarial/property-based testing and large-batch generation analysis, not basic debugging.**
