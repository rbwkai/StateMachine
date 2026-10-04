# DWS-Bench Generator: Complete Implementation Reference

**Repository state described:** 2026-10-04  
**Purpose:** Explain how the generator currently works, the decisions that govern it,
and the places where the implementation still differs from the written contract.

This document is an implementation reference, not a proposal. It distinguishes:

- **Implemented behavior:** what the current Python code actually does.
- **Contract decision:** what `SPEC.md`, `AGENTS.md`, or `documentation/DECISIONS.md`
  says the system must do.
- **Open divergence:** a place where those two are not yet the same.

The governing authority order is:

1. thesis proposal;
2. [`SPEC.md`](../SPEC.md);
3. [`AGENTS.md`](../AGENTS.md);
4. docstrings and comments;
5. executable code.

When this document reports a divergence, it is not silently choosing code over the
contract. It is recording the gap that must be resolved before a dataset is frozen.

---

## 1. Generator purpose

DWS-Bench is a symbolic-first benchmark generator for dynamic world-state tracking.
It creates a sequence of valid world operations, selects or receives a query about
the resulting world, computes symbolic gold answers, renders the operation trace as
natural language, and records enough metadata to audit the generated instance.

The central design decision is:

> The simulator is the source of truth. Natural-language text is a rendering of the
> simulator trace, never a second state machine.

The generator is intended to isolate structural factors such as:

- target-relevant update depth (`T`);
- distractor updates (`D`);
- lifetime entity count (`E`);
- state revision (`V`);
- narrative distractors (`N`);
- canonical update count (`U`);
- rendered length (`L`);
- operation family and structural semantics.

The generator itself does not establish model performance. Model inference and
scoring consume serialized instances later. The current repository has generation
and validation infrastructure, but no current frozen dataset or run results in the
working tree.

---

## 2. Three Research Questions

The generator is designed around three research questions:

### RQ1 — Sequential Dependency and Mutation Structure
How does increasing target-relevant state depth affect the ability of small
instruction-tuned language models to recover the final world state, and does
degradation depend on the type of state mutation rather than depth alone?

**Core experiment:** Vary `T` across 7 mutation structures. Keep `E`, `D`, `N` controlled.

### RQ2 — Interference Decomposition and Supersession
When target-relevant state transitions are held constant, how do state-changing
distractors, narrative distractors, and context length differentially affect
dynamic-state reasoning, and does superseded-information interference produce a
distinct stale-state failure pattern?

**Core experiment:** Fixed target trajectory (interleaved_chain, E=3, T=8). Cross `D × N` with length-matched controls. Supersession via revision family.

### RQ3 — Model Scale and Reasoning Strategy
How do model scale and explicit reasoning prompting alter the accuracy and failure
profile of dynamic-state reasoning, and does the effect of reasoning prompting
depend on mutation structure?

**Core experiment:** 0.5B / 3B / 7B × Direct / Structured on representative RQ1/RQ2 conditions.

---

## 3. Repository layers

The active architecture is split into the following layers.

### 2.1 World simulator

[`world/`](../world/) owns:

- `WorldState`;
- `History`;
- operation validity and application;
- undo/redo stack semantics;
- canonical replay.

It knows nothing about prompts, models, factor sweeps, or natural-language
templates.

### 2.2 Generator

[`generator/`](../generator/) owns:

- family and condition definitions;
- trajectory constructors;
- family-specific validation;
- measured factor computation;
- query and probe construction;
- generation-facing constants.

### 2.3 Renderer

[`render/`](../render/) owns:

- human-readable names;
- operation sentence rendering;
- textual distractor insertion;
- question wording.

The renderer consumes replay-derived information. It must not mutate the symbolic
world or independently decide the gold answer.

### 2.4 Experiment layer

[`experiments/`](../experiments/) turns conditions into batches. It:

1. derives deterministic seeds;
2. constructs a `TrajectorySpec`;
3. builds a trajectory;
4. renders it;
5. measures factors again with rendered sentences;
6. filters or rejects the instance;
7. serializes JSONL records.

### 2.5 Evaluation layer

[`eval/`](../eval/) is downstream. It formats prompts, runs mock or Hugging Face
engines, extracts answers, and scores predictions. It is not part of symbolic
trajectory construction.

---

## 3. World state model

The world state is defined in [`world/state.py`](../world/state.py):

```python
WorldState(
    object_type: Dict[str, str],
    location: Dict[str, str],
    containers: Set[str],
    step_index: int = 0,
)
```

### 3.1 `object_type`

`object_type` maps every entity that has ever existed to its type:

```text
o0 -> key
o1 -> cup
```

This mapping intentionally survives removal. Keeping type metadata allows later
rendering, queries, and diagnostics to refer to an entity that is no longer placed.

### 3.2 `location`

`location` is a partial mapping containing only currently placed entities:

```text
o0 -> c1
```

If an object is removed, its `object_type` entry remains but its `location` entry
is deleted. Therefore:

```python
gold_location(state, obj_id) is None
```

means the object is not currently placed. It does not necessarily mean that the
object was never created.

### 3.3 `containers`

The container set is fixed for a trajectory. Operations may move objects between
containers but do not create or delete containers.

The default number of containers is three, and the minimum is two.

### 3.4 `step_index`

`step_index` advances once for every successfully applied operation, including
`Undo` and `Redo`. It is not the same thing as the number of target-relevant
updates.

### 3.5 History

[`History`](../world/state.py) contains:

```python
undo_stack: List[WorldState]
redo_stack: List[WorldState]
```

History stores complete cloned states, not inverse operation descriptions.

---

## 4. Operation semantics

[`world/operations.py`](../world/operations.py) defines the operation dataclasses
and `apply_op`.

`apply_op` is the only central state-transition dispatcher. Builders, replay, and
probes are expected to use it rather than mutating `WorldState` directly.

### 4.1 `Put`

```text
Put(obj_id, obj_type, container)
```

Valid only when:

- `obj_id` has never appeared in `object_type` or `location`;
- `container` is in the fixed container set.

Effect:

- creates the type entry;
- creates the location entry.

`Put` operations are initialization/setup operations. They are excluded from `U`,
`T`, and `D` under the current factor contract.

### 4.2 `Move`

```text
Move(obj_id, dst)
```

Valid only when:

- the object is currently placed;
- `dst` is a valid container;
- `dst` differs from the current location.

Effect:

- changes exactly one location entry.

### 4.3 `Remove`

```text
Remove(obj_id)
```

Valid only when the object is currently placed.

Effect:

- deletes the object from `location`;
- preserves its `object_type`.

`Remove` remains part of the simulator vocabulary and sampler capability, but
decision D-006 says the benchmark trajectory families do not emit it. If a future
family includes it, that must be an explicit design decision.

### 4.4 `Split`

```text
Split(source_obj_id, new_obj_id)
```

Valid only when:

- the source is currently placed;
- the new id has never existed.

Effect:

- creates the child id;
- copies the source type;
- places the child in the source's current container;
- leaves the source in place.

This changes entity cardinality and is the required structural operation for
`split_chain`.

### 4.5 `Merge`

```text
Merge(src_container, dst_container)
```

Valid only when:

- source and destination differ;
- both containers are valid;
- source is non-empty.

Effect:

- moves every object currently in the source container to the destination.

This is a whole-container relocation, not a single-object move.

### 4.6 `Swap`

```text
Swap(container_a, container_b)
```

Valid only when:

- the containers differ;
- both are valid.

Empty containers are allowed.

Effect:

- atomically exchanges all contents between the two containers.

The implementation first snapshots both contents before writing either side, so
objects are not accidentally moved twice.

### 4.7 `Undo`

`Undo` requires a non-empty undo stack.

Effect:

1. clone the current state onto the redo stack;
2. pop the latest prior state from the undo stack;
3. restore it;
4. increment `step_index`.

### 4.8 `Redo`

`Redo` requires a non-empty redo stack.

Effect:

1. clone the current state onto the undo stack;
2. pop the latest undone state from the redo stack;
3. restore it;
4. increment `step_index`.

### 4.9 New ordinary operations clear redo history

For every non-history operation:

1. validate against the current state;
2. push a clone of the current state to `undo_stack`;
3. clear `redo_stack`;
4. clone and apply the operation;
5. increment `step_index`.

This rule is central to redo-validity probes.

---

## 5. Canonical replay

[`world/replay.py`](../world/replay.py) provides:

```python
replay_trace(
    ops,
    containers,
    history=None,
) -> (trace, final_state, final_history)
```

Replay starts from an empty world and applies operations in order through
`apply_op`.

Each trace element is:

```text
(operation, state_before, state_after)
```

The same trace is intended to feed:

- final-state gold;
- step-wise gold;
- factor measurement;
- narrative rendering;
- counterfactual probes;
- constructor/replay consistency checks.

If an operation is invalid at its position, replay raises `InvalidOperation`.
This is especially important for counterfactual deletion: deleting one earlier
operation can make a later operation invalid, which is a distinct result from a
valid replay whose answer stays the same.

---

## 6. Family taxonomy

[`generator/dataset_spec.py`](../generator/dataset_spec.py) groups the eight
families into five capability groups.

| Capability group | Families | Intended phenomenon |
|---|---|---|
| Sequential state tracking | `basic_chain`, `revision` | ordinary chains and revisits |
| Multi-entity interference | `interleaved_chain` | target updates mixed with other entities |
| Identity transformation | `split_chain`, `merge_chain` | entity creation and whole-container fusion |
| Global state operations | `swap_chain` | bilateral container exchange |
| Temporal edit history | `undo_chain`, `undo_redo_chain` | rollback and reapplication |

The family registry is currently defined in
[`generator/dataset_spec.py`](../generator/dataset_spec.py), but related family
sets also exist in validation and trajectory modules. Centralization remains an
open cleanup item.

---

## 7. Requested condition model

`Condition` in [`generator/dataset_spec.py`](../generator/dataset_spec.py)
represents an experimental request:

- `family`;
- `T`;
- `E`;
- `D`;
- `experiment`;
- `generation_status`.

Derived values:

```text
U = T + D
S = initial_placements + U
```

Initial placements are normally `E`. `split_chain` is special: it initially places
one source entity and creates the second entity later through `Split`.

### 7.1 Numeric checks

The current `Condition` constructor enforces:

- `T >= 1`;
- `E >= 1`;
- `D >= 0`.

The earlier `D < T` restriction was deliberately removed. Conditions with more
distractor updates than target updates are legal.

### 7.2 Generation status

The intended lifecycle is:

```text
PENDING_CALIBRATION
        ↓
PENDING_GENERATION
        ↓
GENERATED
```

`GENERATED` is intended to be immutable. The current working tree has no generated
dataset, so this lifecycle has not yet been exercised for a new release.

---

## 8. `TrajectorySpec`

[`generator/trajectory_specs.py`](../generator/trajectory_specs.py) is the
constructor-level specification. It contains:

- `family`;
- `entity_count`;
- `num_containers`;
- `total_updates`;
- `target_updates`;
- `distractor_updates`;
- `min_interleaving`;
- `revision_count`;
- `min_unique_target_locations`;
- `structural_ops`;
- optional `target_obj`.

### 8.1 Generic checks

It checks:

- family is registered;
- entity count is positive;
- container count meets the minimum;
- total updates are positive;
- target updates are positive;
- distractor updates are non-negative;
- interleaving is in `[0, 1]`;
- revision count is non-negative;
- minimum unique target locations is positive.

### 8.2 Ordinary update-count consistency

For ordinary families, it requires:

```text
total_updates = target_updates + distractor_updates
```

For structural families, this check is skipped because the current constructors
interpret `target_updates` using family-specific structural conventions.

### 8.3 Current schema inconsistency

The written decisions originally moved toward one contract and removed the
v1/v2 duality. However, the active file currently exposes a transitional
`schema_version` property that returns `"v1"` and does not accept
`schema_version` as a dataclass constructor field.

At the same time:

- `generate.py` advertises `--schema-version`;
- experiment code passes `schema_version`;
- trajectory constructors contain schema-dependent branches;
- tests expect `TrajectorySpec(..., schema_version="v2")` to fail.

Therefore, the intended schema behavior and active constructor behavior are not
fully reconciled. This must be resolved before relying on schema-specific
generation.

---

## 9. Trajectory construction

[`generator/trajectories.py`](../generator/trajectories.py) is the main constructor
module.

Every helper routes operations through `apply_op`. The general pattern is:

1. create an empty world with fixed containers;
2. add setup `Put` operations;
3. add family-specific target and distractor operations;
4. maintain the constructor's current state/history;
5. return a `ConstructedTrajectory`.

`ConstructedTrajectory` contains:

- `ops`;
- `containers`;
- `final_state`;
- `history`;
- `target_obj`;
- `spec`;
- optional measured factors.

### 9.1 `basic_chain`

Shape:

```text
Put(target)
Move(target)
Move(target)
...
```

Properties:

- exactly one entity;
- no distractor updates;
- every post-setup operation moves the target;
- each destination differs from the current destination.

### 9.2 `interleaved_chain`

Shape:

```text
Put(target)
Put(distractor_1)
Put(distractor_2)
...
interleaved target Moves and distractor Moves
```

Properties:

- at least two entities;
- at least one target update;
- at least one distractor update;
- target and distractor operations are deliberately interleaved;
- validation measures whether distractors occur between target updates.

### 9.3 `revision`

Shape:

```text
Put(target)
target Moves with at least one revisit
```

The destination pattern is designed to revisit a previous location after an
intervening different location. The family requires at least three target
updates, but `revision_count` is not currently enforced as an exact requested
factor.

### 9.4 `split_chain`

The family creates a child through `Split`.

The intended corrected design queries the spawned child so that the split itself
is causally meaningful:

```text
Put(source)
source movement
Split(source, child)
...
```

The legacy and intended corrected paths differ. The current schema inconsistency
means callers must not assume the corrected branch is reachable without first
resolving `TrajectorySpec`.

### 9.5 `merge_chain`

The target is placed in a source container and a `Merge` moves the source
container's contents into another container.

The corrected structural design places `Merge` as the last target-affecting
operation, preventing a later ordinary target move from overwriting the answer.

### 9.6 `swap_chain`

The target and a companion are placed in participating containers. `Swap`
exchanges their contents.

The corrected structural design places `Swap` at the end of target-affecting
operations so the final answer depends on the exchange rather than a later
ordinary `Move`.

### 9.7 `undo_chain`

The corrected design ends with `Undo`:

```text
target moves
Undo
```

This keeps rollback visible in the final answer.

### 9.8 `undo_redo_chain`

The corrected design reserves history operations near the end:

```text
target Move
Undo
Redo
```

Both `Undo` and `Redo` must be causally relevant under the structural rule.

---

## 10. Public construction gate

`build_trajectory(...)` is intended to be the only normal constructor entry point.

The intended gate is:

```text
family constructor
    ↓
canonical replay
    ↓
constructor/replay state equality
    ↓
family validation
    ↓
factor measurement
    ↓
rendering
    ↓
rendered-factor verification
```

The current implementation performs the following inside or around the build path:

1. selects the family constructor;
2. constructs operations;
3. replays them;
4. compares final location, object-type, and container data;
5. applies an inline structural necessity check in relevant paths;
6. calls `validate_trajectory`;
7. attaches measured factors.

### 10.1 Important current gap

`build_trajectory` measures factors but does not universally call:

- `verify_factors`;
- `verify_length`.

The experiment pipeline performs manual `E/T/D` equality checks and optional `V`
checks after rendering. The lightweight CLI path does not currently provide the
same complete verification gate.

This is a release blocker for a strict frozen-data workflow.

---

## 11. Family-specific validation

[`generator/trajectory_validation.py`](../generator/trajectory_validation.py)
checks family invariants after construction.

### 11.1 Common checks

Validation checks include:

- operation sequence is non-empty;
- target exists in the expected identity set;
- expected setup entity count;
- no duplicate entity creation;
- required operation counts;
- target and distractor counts;
- unsupported operation rejection.

### 11.2 Basic validation

Requires:

- one entity;
- zero distractor updates;
- every move targets the queried entity.

### 11.3 Interleaving validation

Requires:

- at least two entities;
- target moves;
- distractor moves;
- distractors between target moves;
- `min_interleaving` threshold.

The interleaving score is approximately:

```text
distractor moves between consecutive target moves
------------------------------------------------
total move operations
```

### 11.4 Revision validation

Requires:

- one entity;
- no distractor updates;
- at least three target updates;
- enough unique target locations;
- a genuine revisit after an intervening location.

### 11.5 Structural validation

Structural validation checks required operation types and family-specific identity
constraints. Split has special handling because the queried target may be the
spawned child in the corrected design.

---

## 12. Structural causality

Structural families declare required operations:

| Family | Required operation |
|---|---|
| `split_chain` | `Split` |
| `merge_chain` | `Merge` |
| `swap_chain` | `Swap` |
| `undo_chain` | `Undo` |
| `undo_redo_chain` | `Undo` and `Redo` |

The intended decision is D-004:

1. required operation is present;
2. it affects the queried target or creates the queried identity;
3. deleting it either changes the final answer or invalidates replay;
4. unrelated structural operations do not have to be causal;
5. a final ordinary target `Move` cannot be the only determinant of the answer.

### 12.1 Duplicate implementation warning

[`generator/structural.py`](../generator/structural.py) contains a reusable
structural-causality validator. The trajectory build path also contains inline
structural checks. These are not yet guaranteed to be identical or centrally
wired.

Before freezing data, one implementation should become the authoritative gate and
the other should be removed or made a thin wrapper.

---

## 13. Factor measurement

[`generator/metadata.py`](../generator/metadata.py) is the intended single owner of
factor definitions.

`MeasuredFactors` contains:

- `E_actual`;
- `T_actual`;
- `D_actual`;
- `V_actual`;
- `L_word`;
- `N_actual`.

Derived:

```text
U_actual = T_actual + D_actual
L_actual = L_word
```

### 13.1 `E`: lifetime entities

Count the unique ids created by:

- `Put.obj_id`;
- `Split.new_obj_id`.

The implementation also folds in `Split.source_obj_id`; because the source must
already have existed, this normally does not change the union.

### 13.2 `T` and `D`: state-based target effect

For every post-setup operation, compare the target location before and after:

```text
affects_target =
    location_before[target] != location_after[target]
```

If true, count the operation in `T`; otherwise count it in `D`.

This means `T` can include:

- target `Move`;
- `Split` creating or changing the target;
- `Merge` moving the target;
- `Swap` moving the target;
- target-affecting `Undo`;
- target-affecting `Redo`;
- target `Remove`, because location changes to `None`.

This is deliberately state-based rather than a syntactic operation whitelist.

### 13.3 `U`

```text
U = T + D
```

All non-`Put` operations are classified into exactly one of these two groups.

### 13.4 `V`: revisions

Current definition:

```text
V =
    target-location revisits
    + target-affecting Undo/Redo count
```

A revisit requires returning to a previous destination after an intervening
different location.

Known unresolved behavior: one target-affecting `Undo` or `Redo` can be counted
both as a revisit and as a history reversal. This double count is pinned by tests
and remains an open contract decision.

### 13.5 `N`: narrative distractors

`N` counts pure-text distractor sentences that perform no symbolic transition.

If explicit rendered sentences are available, measurement can derive `N` from
the supplied count or from the difference between sentence count and operation
count. If sentences are not supplied, the explicit count is retained.

### 13.6 `L`: rendered word count

`L_word` is:

```python
sum(len(sentence.split()) for sentence in sentences)
```

The generation ceiling is `L_MAX_WORDS = 600`.

The decision log separates:

- `L_word`: generation gate;
- `L_tok`: tokenizer diagnostic for downstream evaluation.

The generator does not require `transformers` and does not use tokenizer length
as its primary gate.

### 13.7 Verification

`verify_factors` checks exact equality for:

- requested `E` vs `E_actual`;
- requested `T` vs `T_actual`;
- requested `D` vs `D_actual`.

`V` is checked only when a caller supplies `min_v` or `intended_v`.

`verify_length` checks the word ceiling, but an unrendered sentinel length of `-1`
can pass unless the caller renders first. The release pipeline must therefore
verify length only after rendering.

---

## 14. Query construction

[`generator/probes.py`](../generator/probes.py) defines:

- `LocationQuery`;
- `CountQuery`;
- `RedoValidityQuery`.

### 14.1 Location query

Reads `state.location.get(obj_id)`.

The result is a container id or `None`.

### 14.2 Count query

Reads the number of objects of a type currently in a container.

### 14.3 Redo-validity query

Reads history state rather than `WorldState`:

```text
can_redo(history)
```

This is why redo-validity records must preserve the final history object or an
equivalent derived label.

### 14.4 Candidate selection

Candidate queries can include:

- location queries for known entity ids;
- count queries for container/type combinations.

`select_query` can use `analysis/query_analysis.py` to filter candidates based on
measured or syntactically inferred properties such as relevance, dependency depth,
interleaving, revision, and answer changes.

### 14.5 Relevance divergence

There are two notions of relevance:

1. `metadata.py`: actual before/after target-state change;
2. `query_analysis.py`: syntactic operation/query relationship.

They can disagree. The decision log keeps this open because changing either one
would affect query selection and factor interpretation.

---

## 15. Counterfactual probes

The current probe system classifies every candidate operation removal into:

1. `valid_answer_changing`;
2. `valid_answer_preserving`;
3. `invalid_replay`;
4. `excluded_setup`.

For each candidate:

1. remove exactly one operation;
2. replay the reduced sequence;
3. classify invalid replay separately;
4. compare final query answers when replay succeeds;
5. record the operation index and answer outcome.

Setup `Put` operations are excluded by default because deleting setup changes the
initial conditions rather than intervening on a post-initialization event.

The accounting object asserts that all candidates are represented by exactly one
bucket.

The selected probe set can request a balance between valid answer-changing and
valid answer-preserving probes. If one class is unavailable, the degeneracy is
reported rather than silently presented as balanced.

A valid `None` answer for a removed target is distinct from an invalid replay.

### 15.1 Redo probes

Redo-validity examples are deliberately constructed in two classes:

- valid redo: perform `Undo` and stop;
- invalid redo: perform `Undo`, then a new ordinary operation that clears redo.

The label is read from `can_redo(history)`, not assumed from the requested class.

---

## 16. Rendering

The active renderer is [`render/narrative.py`](../render/narrative.py).

It:

1. replays the operation sequence;
2. renders one sentence per operation;
3. uses the corresponding `state_before` for source-sensitive wording;
4. returns sentences and final state.

### 16.1 Names

[`render/names.py`](../render/names.py) provides deterministic names using a
seeded local random generator.

Repeated object types receive distinct surface names using ordinal/duplicate
forms.

### 16.2 Move wording

The corrected design uses destination-only Move wording to reduce source-location
shortcuts:

```text
The object was moved to container B.
```

The decision log says this wording should apply under one contract, not only to a
schema-specific family. The active code still contains transition branches, so
this must be verified before generation.

### 16.3 Text distractors

Textual distractors are inserted into the sentence sequence at seeded positions.
They:

- increase narrative length;
- increase `N`;
- do not mutate symbolic state;
- do not alter symbolic gold.

### 16.4 Duplicate renderer

[`render/templates.py`](../render/templates.py) remains as a second, overlapping
renderer surface. The active paths generally use `narrative.py`, but the duplicate
implementation creates maintenance and vocabulary-drift risk.

---

## 17. Experiment generation pipeline

[`experiments/_common.py`](../experiments/_common.py) is the main batch path.

For each requested instance it:

1. derives a deterministic seed;
2. creates separate trajectory and naming RNG streams;
3. creates a `TrajectorySpec`;
4. calls `build_trajectory`;
5. assigns names;
6. renders the canonical trace;
7. inserts optional text distractors;
8. measures factors again using rendered sentences;
9. checks requested/measured `E`, `T`, and `D`;
10. checks optional `V` requirements;
11. serializes the canonical trace;
12. computes a trace hash;
13. computes step-wise gold;
14. creates question and final answer fields;
15. emits a JSON-serializable record.

Failed attempts are retried with derived seeds. After retry exhaustion, the
instance fails rather than silently emitting an invalid record.

### 17.1 Record fields

Records can contain:

- `schema_version`;
- `instance_id`;
- `family`;
- `experiment`;
- `condition_id`;
- `seed`;
- `trace_hash`;
- `requested_factors`;
- `measured_factors`;
- `spec`;
- `canonical_trace`;
- `sentences`;
- `context`;
- `question`;
- `query_entity`;
- `gold_container`;
- `gold_answer`;
- `step_wise_gold`;
- `step_wise_gold_answers`;
- `final_state`.

The exact active schema still depends on the schema-version inconsistency described
earlier.

---

## 18. Experiment families and grids

The scripts currently define these planned sweeps for the three research questions.

### 18.1 RQ1 — Mutation & Depth

[`experiments/rq1_mutation_depth.py`](../experiments/rq1_mutation_depth.py):

| Family | E | D | N | T levels | Instances/cond |
|---|---|---|---|---|---|
| basic_chain | 1 | 0 | 0 | {2, 4, 6, 8, 12, 16} | 50 |
| revision | 1 | 0 | 0 | {4, 8, 12, 16} | 50 |
| split_chain | 2 | 0 | 0 | {4, 8, 12, 16} | 50 |
| merge_chain | 2 | 0 | 0 | {4, 8, 12, 16} | 50 |
| swap_chain | 2 | 0 | 0 | {4, 8, 12, 16} | 50 |
| undo_chain | 1 | 0 | 0 | {4, 8, 12, 16} | 50 |
| undo_redo_chain | 1 | 0 | 0 | {4, 8, 12, 16} | 50 |

Total: ~1,500 instances.

### 18.2 RQ2 — Interference & Supersession

[`experiments/rq2_interference.py`](../experiments/rq2_interference.py):

**Main D×N grid** (fixed target: interleaved_chain, E=3, T=8):

| Condition | D | N | Instances |
|---|---|---|---|
| D=4, N=0 | 4 | 0 | 50 |
| D=8, N=0 | 8 | 0 | 50 |
| D=16, N=0 | 16 | 0 | 50 |
| D=4, N=4 | 4 | 4 | 50 |
| D=4, N=8 | 4 | 8 | 50 |
| D=4, N=16 | 4 | 16 | 50 |

**Supersession** (revision family, same T=8 target trajectory):

| Condition | Family | E | T | V floor |
|---|---|---|---|---|
| revision T=8 | revision | 1 | 8 | V_actual ≥ 2 |

Total: ~700 instances.

### 18.3 RQ3 — Scale & Reasoning

[`experiments/rq3_scale_reasoning.py`](../experiments/rq3_scale_reasoning.py):

Evaluation-only (consumes RQ1/RQ2 data). No data generation.

| Model | Prompt | Conditions |
|---|---|---|
| qwen2.5-0.5b | direct | Representative RQ1/RQ2 conditions |
| qwen2.5-0.5b | structured | Representative RQ1/RQ2 conditions |
| qwen2.5-3b | direct | Representative RQ1/RQ2 conditions |
| qwen2.5-3b | structured | Representative RQ1/RQ2 conditions |
| qwen2.5-7b | direct | Representative RQ1/RQ2 conditions |
| qwen2.5-7b | structured | Representative RQ1/RQ2 conditions |

---

## 19. CLI generation path

[`generate.py`](../generate.py) provides lightweight family examples and exposes:

- family;
- count;
- seed;
- output path;
- advertised schema version.

This path is not equivalent to the experiment pipeline. It currently does not
attach the same measured-factor record, does not apply the full rendered-length
gate, and has compatibility issues around passing `schema_version` into the
current `TrajectorySpec`.

For benchmark release generation, the experiment pipeline and a strict freeze
check should be preferred over the lightweight example CLI.

---

## 20. Determinism

The design requires:

- explicit `random.Random` instances;
- no module-global random calls;
- deterministic seed derivation;
- deterministic name assignment;
- no timestamps or UUIDs in instance content;
- stable serialization;
- trace hashes for diversity and identity checks.

The experiment layer separates trajectory RNG from naming RNG so changing surface
names does not silently change symbolic trajectory construction.

The intended reproducibility property is:

```text
same condition + same seed + same code/spec
    -> same symbolic trace and same serialized instance bytes
```

Changing RNG consumption, operation wording, gold extraction, or factor semantics
requires a contract/version decision and regeneration.

---

## 21. Dataset-level release gates

Before a condition can move from pending generation to generated, the release
process should verify:

1. all reachability probes pass;
2. every trajectory replays successfully;
3. constructor and replay final states agree;
4. requested `E/T/D` equal measured `E/T/D`;
5. required `V` floors or exact values pass;
6. rendered `L_word <= 600`;
7. structural operations are causally necessary;
8. no forbidden trailing target Move hides structural behavior;
9. seeds are unique;
10. trace hashes meet the distinctness threshold;
11. probe classes are balanced or the imbalance is explicitly recorded;
12. record serialization is deterministic;
13. no secrets or local machine paths enter artifacts.

The old benchmark and result outputs were removed by decision D-001/D-010. They
must not be treated as evidence for the current generator.

---

## 22. Tests

Relevant test surfaces include:

- [`test/smoke_test_trajectories.py`](../test/smoke_test_trajectories.py):
  all eight families, invalid specs, replay consistency, determinism;
- [`test/test_invariants.py`](../test/test_invariants.py):
  randomized invariant and trace-diversity checks;
- [`test/test_measured_factors.py`](../test/test_measured_factors.py):
  factor measurement and mismatch detection;
- [`test/test_factor_contract.py`](../test/test_factor_contract.py):
  condition rules, state-based classification, length checks, and RNG separation;
- [`test/test_probes.py`](../test/test_probes.py):
  counterfactual accounting and redo labels;
- [`test/test_scoring.py`](../test/test_scoring.py):
  downstream extraction and protocol behavior.

The current repository state is not fully green:

- the deleted datasets cause the CLI evaluation test to fail because it expects
  `data/rq1_depth/rq1_depth.jsonl`;
- the master runner has a known `hashlib` import issue in
  [`test/test_invariants.py`](../test/test_invariants.py).

These are repository-state/test integration issues, not evidence that the
generator release is ready.

---

## 23. Decisions of record

### D-001: pause freezing

No new dataset is frozen until the factor contract is locked and a small pilot is
inspected. Old data and results are not reused as current evidence.

### D-002: two length quantities

`L_word` controls generation. Tokenizer length is a downstream diagnostic.

### D-003: one contract

The intended end state removes v1/v2 behavioral branching. Existing code still
contains a transitional schema mismatch and must be reconciled.

### D-004: required structural operations

Structural causality is declared by family. It is not inferred from every
structural operation being relevant.

### D-005: one factor owner

`generator/metadata.py` owns factor definitions. Requested factors must be checked
against replay-measured values.

### D-006: `Remove` remains outside families

The simulator supports `Remove`; the current benchmark families do not emit it.

### D-007: probe accounting

Every candidate removal belongs to an explicit accounting class. Invalid replay is
not silently discarded.

### D-008: legacy implementations

The decision log says root-level legacy modules were removed. The current working
tree still contains some legacy files, so the decision and filesystem are not yet
fully synchronized.

### D-009: retain all planned family sweeps

No family or factor is removed merely to make generation pass.

### D-010: data/results are regenerated

Datasets and results are not committed as current evidence. They are generated
from recorded conditions and seeds after the contract is locked.

### D-011: central constants

Thresholds, limits, and version stamps belong in
[`generator/constants.py`](../generator/constants.py).

---

## 24. Open implementation issues before freezing

The following issues must be resolved or explicitly accepted in a freeze decision:

1. Reconcile the schema-version property, constructor calls, and tests.
2. Wire one structural-causality validator into the normal build gate.
3. Make `verify_factors` and `verify_length` unavoidable after rendering.
4. Define one interpretation of structural `target_updates` and `total_updates`.
5. Decide whether `revision_count` means an exact `V` target or only a descriptive
   request.
6. Resolve the `V` double-count for target-affecting history operations.
7. Reconcile state-based factor relevance with syntactic query analysis.
8. Centralize family registries.
9. Centralize object-type vocabulary.
10. Remove or explicitly quarantine duplicate renderer and legacy module surfaces.
11. Repair tests that assume deleted data without weakening their purpose.
12. Run a seed sweep and inspect generated records by hand.
13. Generate a small pilot before any large immutable release.

---

## 25. End-to-end summary

The intended complete path is:

```text
Condition
  ↓
TrajectorySpec
  ↓
family constructor
  ↓
apply_op for every operation
  ↓
canonical replay trace
  ↓
constructor/replay equality
  ↓
family validation
  ↓
measured E/T/D/V
  ↓
required factor verification
  ↓
replay-driven query and step-wise gold
  ↓
replay-driven rendering
  ↓
text distractor insertion
  ↓
rendered L/N measurement
  ↓
length and release gates
  ↓
counterfactual and redo probes
  ↓
deterministic JSONL record
```

The most important engineering invariant is that no later stage is allowed to
invent symbolic state. If a sentence, answer, factor, or probe cannot be traced
back to the canonical replay, the instance is not compliant with the generator
contract.
