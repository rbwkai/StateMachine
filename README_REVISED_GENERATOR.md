# DWS-Bench Revised Generator

## 1. Overview

DWS-Bench is a deterministic benchmark for evaluating whether a language model can maintain and update a changing world state from a sequence of natural-language events.

The benchmark is built around a simple principle:

> The simulator is the source of truth. Natural-language text is only a rendering of the simulator's canonical operation trace.

The revised generator changes the generation philosophy from:

```text
random valid trajectory
        ↓
heuristic query
```

to:

```text
sample valid trajectory
        ↓
analyze trajectory
        ↓
select a query satisfying an explicit QuerySpec
        ↓
build canonical gold trajectory
        ↓
render operations as natural language
        ↓
optionally inject distractors
        ↓
optionally generate counterfactual probes
        ↓
export benchmark instance
```

This makes difficulty properties explicit and measurable rather than depending on accidental properties of randomly generated examples.

---

## 2. Repository Structure

The current repository is organized as follows:

```text
STATEMACHINE/
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
├── world/
│   ├── __init__.py
│   ├── errors.py
│   ├── operations.py
│   ├── replay.py
│   └── state.py
│
├── analysis.py
├── example.py
├── generator.py
├── pipeline.py
├── sampler.py
├── trajectory.py
│
├── .gitignore
│
├── generator_renderer_implementation.md
└── README_REVISED_GENERATOR.md
```

There are also `__pycache__` directories created automatically by Python. They are implementation artifacts and should not be committed to Git.

---

# 3. Architectural Layers

The repository has four logical layers.

## Layer 1 — World simulator

Located in:

```text
world/
```

This layer defines the actual state of the world and the operations that can change it.

It knows nothing about natural-language rendering, benchmark difficulty, or LLMs.

---

## Layer 2 — Trajectory generation

Located mainly in:

```text
generator/sampler.py
sampler.py
```

This layer creates sequences of valid operations.

It is responsible for satisfying the basic structural budgets:

- `E` — entity/object budget
- `U` — update/operation count
- available operation vocabulary
- number of containers

---

## Layer 3 — Query and trajectory analysis

Located in:

```text
analysis.py
generator/probes.py
trajectory.py
```

This layer determines whether a generated trajectory actually has the properties required by a benchmark condition.

Examples:

- Does the queried object change location?
- How many operations are relevant?
- How many world-state changes occur?
- How deep is the dependency chain?
- How much irrelevant interleaving exists?
- Does the query require revision?
- Is the final answer sensitive to deletion of a particular operation?

---

## Layer 4 — Natural-language rendering

Located in:

```text
render/
```

This layer converts the canonical symbolic operation sequence into natural-language sentences.

It does not decide the gold answer.

It does not modify the world.

It does not determine trajectory validity.

Its job is surface realization only.

---

# 4. The World Model

The central state representation is:

```python
@dataclass
class WorldState:
    object_type: Dict[str, str]
    location: Dict[str, str]
    containers: Set[str]
    step_index: int = 0
```

For example:

```text
object_type:
    o0 -> candle
    o1 -> map

location:
    o0 -> c1
    o1 -> c0

containers:
    c0, c1, c2
```

`object_type` records the type of every object that has existed.

`location` records only objects that are currently placed.

Therefore, an object can remain in `object_type` after being removed while disappearing from `location`.

This is intentional.

For example:

```text
Put(o0, candle, c0)
Remove(o0)
```

produces:

```text
object_type:
    o0 -> candle

location:
    {}
```

This allows the renderer and analysis code to continue referring to the object after removal.

---

# 5. History and Undo/Redo

Undo and redo require information beyond the current `WorldState`.

The repository therefore uses:

```python
@dataclass
class History:
    undo_stack: List[WorldState]
    redo_stack: List[WorldState]
```

The important invariant is:

```text
normal operation
    → save previous state on undo_stack
    → clear redo_stack

Undo
    → move current state to redo_stack
    → restore previous state

Redo
    → move current state to undo_stack
    → restore redo state
```

A new operation after an `Undo` clears the redo stack.

For example:

```text
A
B
Undo
C
```

means the old `B` state cannot be redone after `C`.

This behavior is used by the dedicated redo-validity probe.

---

# 6. Supported Operations

The simulator currently supports:

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

## Put

Creates a new object in a container.

```text
Put(o0, candle, c0)
```

Natural-language rendering:

```text
A candle was placed in the blue closet.
```

---

## Move

Changes the location of an existing object.

```text
Move(o0, c1)
```

Rendering:

```text
The candle was moved from the blue closet to the old shelf.
```

---

## Remove

Removes an object from its current container.

```text
Remove(o0)
```

The object remains known by `object_type`, but it no longer has a location.

---

## Split

Creates a second object with the same type as an existing object.

```text
Split(o0, o1)
```

This increases the number of entities in the world.

---

## Merge

Moves every object from one container to another.

```text
Merge(c0, c1)
```

This is a broad state transformation because multiple objects may change location simultaneously.

---

## Swap

Exchanges the contents of two containers.

```text
Swap(c0, c1)
```

This can simultaneously move multiple objects.

---

## Undo

Reverts the most recent normal operation.

---

## Redo

Reapplies the most recently undone operation, if the redo stack is still valid.

---

# 7. `world/operations.py`

This file contains the operation dataclasses and the central `apply_op()` dispatcher.

The most important design decision is that all state transitions pass through:

```python
apply_op(op, state, history)
```

This makes it the single authority for:

- operation validity
- state mutation
- history updates
- undo-stack management
- redo-stack management

The generator, replay system, and counterfactual evaluator therefore share exactly the same transition semantics.

This prevents different parts of the benchmark from accidentally implementing different versions of the world.

---

# 8. `world/replay.py`

`replay_trace()` replays an operation sequence from an empty world.

It returns:

```python
(
    trace,
    final_state,
    final_history,
)
```

where every trace entry is:

```python
(
    operation,
    state_before,
    state_after,
)
```

This shared trace is important.

The renderer uses it to know what an operation should say.

The trajectory builder uses it to calculate step-wise answers.

The counterfactual system uses the same replay semantics to evaluate interventions.

Therefore:

```text
operation
    ↓
apply_op()
    ↓
state_before / state_after
    ├── renderer
    ├── trajectory gold
    ├── analysis
    └── counterfactual evaluation
```

There is no independent "text state" that can drift away from the simulator.

---

# 9. `generator/sampler.py`

The sampler generates valid operation sequences.

Instead of sampling arbitrary operations and repeatedly discovering that they are invalid, it first determines which operation types are currently feasible.

For example:

```python
Move
```

is only feasible when at least one object is currently located somewhere.

Similarly:

```python
Undo
```

requires a non-empty undo stack.

```python
Redo
```

requires a non-empty redo stack.

The sampler therefore follows:

```text
current state
      ↓
determine feasible operation types
      ↓
sample an operation type
      ↓
construct a concrete valid operation
      ↓
apply operation
      ↓
repeat
```

This considerably reduces invalid-operation rejection.

---

# 10. Entity Budget `E`

`entity_count` controls how many objects can be created.

Objects are created primarily through:

```text
Put
Split
```

The sampler tracks:

```python
objects_created
```

and prevents creation beyond the configured entity budget.

Therefore `E` controls the potential entity capacity of an instance rather than simply counting arbitrary mentions in the final text.

---

# 11. Update Budget `U`

`update_count` controls the number of canonical operations.

For example:

```text
U = 8
```

means the canonical trajectory contains eight operations.

Important:

> Distractor sentences do not count toward `U`.

If:

```text
U = 8
N = 5
```

the model may see 13 sentences, but only 8 correspond to state-changing operations.

This separation is necessary if `N` is intended to measure textual interference independently of simulator complexity.

---

# 12. Distractor Budget `N`

Distractors are generated in:

```text
render/names.py
render/narrative.py
```

They are pure text.

They do not enter:

```python
ops_applied
```

They do not enter:

```python
trajectory
```

They do not enter:

```python
WorldState
```

They are inserted only after the canonical trajectory has been generated.

The flow is:

```text
canonical operation sentences
            +
       distractors
            ↓
     splice_distractors()
            ↓
      final sentences
```

Therefore:

```text
N changes what the model reads
but does not change the gold world state.
```

This is the intended interpretation of the distractor factor.

---

# 13. `analysis.py`

`analysis.py` contains the structural analysis layer.

The central abstraction is:

```python
QuerySpec
```

A `QuerySpec` describes the required difficulty characteristics of a query.

Examples include:

```python
query_type="location"
must_change_from_initial=True
min_relevant_steps=4
min_state_changes=3
min_dependency_depth=4
min_interleaving=0.4
require_revision=True
```

The sampled trajectory is analyzed before being accepted.

---

# 14. Why `QuerySpec` Exists

A purely random generator does not guarantee meaningful difficulty.

For example, suppose:

```text
U = 8
```

but the queried object is placed at step 1 and never touched again.

The example technically has eight updates, but the model only needs to remember one relevant event.

That is not an eight-step state-tracking problem.

`QuerySpec` fixes this by requiring measured properties of the query trajectory.

The benchmark therefore asks:

> Does this sampled trajectory actually instantiate the condition we intended to generate?

rather than:

> Did we happen to sample eight operations?

---

# 15. Query Selection

`generator/probes.py` contains:

```python
candidate_queries()
select_query()
```

The process is:

```text
sample trajectory
       ↓
generate candidate queries
       ↓
analyze each candidate
       ↓
QuerySpec.matches(analysis)
       ↓
accept first valid query
```

If no candidate query satisfies the specification:

```text
reject trajectory
```

The pipeline then samples another trajectory.

This is rejection sampling over trajectories.

---

# 16. Query Types

The current normal query types are:

## Location query

```python
LocationQuery(obj_id)
```

Question:

```text
Where is the candle now?
```

Gold answer:

```text
c1
```

or:

```text
None
```

if the object is not currently placed.

---

## Count query

```python
CountQuery(container, obj_type)
```

Question:

```text
How many candles are in the old shelf now?
```

Gold answer:

```text
2
```

---

## Redo-validity query

```python
RedoValidityQuery
```

is a special marker query.

It is not evaluated by projecting a normal question from `WorldState`.

Instead, redo validity is evaluated from:

```python
History.redo_stack
```

This is why redo-validity generation is handled separately from normal query selection.

---

# 17. `trajectory.py`

`trajectory.py` builds the canonical step-wise gold trajectory.

For every operation, it records information such as:

```text
step
operation
sentence
state_before
state_after
answer_before
answer_after
state_changed
query_changed
```

Conceptually:

```text
operation 0
    ↓
state S0 → S1
query(S0) → query(S1)

operation 1
    ↓
state S1 → S2
query(S1) → query(S2)

...
```

This allows the benchmark to distinguish:

```text
world-state changes
```

from:

```text
query-answer changes
```

These are not the same quantity.

An operation can change the world without changing the answer to the selected query.

---

# 18. `world_state_change_count`

The trajectory summary uses:

```python
world_state_change_count
```

This counts how many operations changed the world state.

For example:

```text
8 operations
8 world-state changes
```

does not imply:

```text
8 query-answer changes
```

A location query might change only twice.

This distinction is important for analyzing benchmark difficulty.

---

# 19. `query_change_count`

`query_change_count` counts the number of steps at which the selected query's answer changes.

Example:

```text
step 0: c0 → c1   changed
step 1: c1 → c1   unchanged
step 2: c1 → c1   unchanged
step 3: c1 → c0   changed
```

Then:

```text
query_change_count = 2
```

The summary can also record:

```text
first_query_change = 0
last_query_change = 3
```

This identifies when the query becomes relevant and when its final answer is established.

---

# 20. Counterfactual Probes

Counterfactual probes test causal sensitivity.

For each candidate operation:

```text
remove operation i
        ↓
replay remaining trajectory
        ↓
compute final query answer
        ↓
compare with original answer
```

The result contains:

```json
{
  "remove_step": 4,
  "original_answer": "c1",
  "counterfactual_answer": "c0",
  "answer_changed": true
}
```

The interpretation is:

```text
answer_changed = true
```

means removing that operation changes the final answer.

Therefore the operation is causally relevant to the final query answer under this deletion intervention.

---

# 21. Counterfactual Selection Policy

`build_counterfactual_probes()` evaluates every possible single-operation deletion.

It separates them into:

```text
sensitive
    answer_changed == true

insensitive
    answer_changed == false
```

It then prefers:

1. one sensitive probe
2. one insensitive probe
3. additional probes if available

This is more informative than randomly selecting deletion points.

A benchmark instance can therefore contain both:

```text
causally necessary operation
```

and:

```text
causally unnecessary operation
```

---

# 22. Invalid Counterfactuals

A deletion can make a later operation invalid.

Example:

```text
Put(o0)
Move(o0, c1)
Remove(o0)
```

If the first `Put` is deleted, the later `Move` may no longer be valid.

The current policy is conservative:

```text
invalid counterfactual → exclude it
```

It does not attempt to repair the remaining trajectory.

This is intentional.

Repairing the trajectory would introduce a second intervention and make causal interpretation less clean.

---

# 23. Natural-Language Rendering

The rendering layer is in:

```text
render/
```

The main file is:

```text
render/narrative.py
```

It contains one rendering function per operation type.

For example:

```text
Put
    ↓
"A candle was placed in the blue closet."

Move
    ↓
"The candle was moved from the blue closet to the old shelf."

Remove
    ↓
"The candle was taken out of the old shelf."

Undo
    ↓
"That last action was undone."
```

The renderer receives:

```python
(operation, state_before, names)
```

This is important for operations such as `Move`, because the source container must be obtained from the state before the operation.

---

# 24. Name Registry

`render/names.py` contains:

```python
NameRegistry
```

The simulator uses symbolic IDs:

```text
c0
c1
c2
o0
o1
```

The renderer maps them to surface names:

```text
c0 → the blue closet
c1 → the old shelf
c2 → the narrow basket
```

The benchmark question does not expose the internal object ID.

Instead:

```text
o0 → candle
```

becomes:

```text
Where is the candle now?
```

This keeps the task closer to natural language entity tracking.

---

# 25. Distractors

Distractors are generated after the canonical trajectory.

Examples:

```text
The blue closet has a faint smell of cedar.

Someone mentioned that maps have become harder to find lately.
```

They are intentionally not operations.

Therefore the model must distinguish:

```text
state-changing event
```

from:

```text
irrelevant textual information
```

without changing the simulator's ground truth.

---

# 26. `pipeline.py`

`pipeline.py` is the main orchestration layer.

Normal generation follows:

```text
1. sample_sequence()
2. select_query()
3. render_narrative()
4. verify replay consistency
5. build_trajectory_and_gold()
6. verify final answer
7. generate distractors
8. construct benchmark record
9. generate counterfactual probes
10. run consistency checks
11. return record
```

If any constraint fails, the trajectory is rejected and another attempt is sampled.

The default is:

```python
max_trajectory_attempts = 200
```

This is why generation can fail with an error such as:

```text
could not generate a trajectory satisfying QuerySpec(...)
after 200 attempts
```

That error does not necessarily mean the code is broken.

It can mean the requested condition is rare or impossible under the current operation vocabulary and budgets.

---

# 27. Why Rejection Sampling Is Used

Suppose the requested condition is:

```python
min_relevant_steps=5
min_dependency_depth=4
require_revision=True
```

A random trajectory may fail these conditions.

Instead of weakening the condition, the generator rejects the trajectory.

This preserves the semantic meaning of the benchmark factor.

The tradeoff is generation efficiency.

Rare conditions may require many attempts.

---

# 28. Dedicated Redo-Validity Generation

Redo validity is sufficiently specialized that it has its own constructor:

```python
build_redo_validity_example()
```

The intended structure is:

```text
normal operations
      ↓
setup operation
      ↓
Undo
      ↓
new operation
      ↓
redo stack invalidated
```

The final benchmark question is:

```text
If someone tried to redo the last undone action right now,
would that succeed?
```

The expected answer is determined from:

```python
can_redo(history)
```

The actual `Redo` operation is not applied.

This tests whether the model understands history semantics rather than simply predicting a location.

---

# 29. Recommended Benchmark Conditions

## A. Basic sequential state tracking

Use:

```python
QuerySpec(
    query_type="location",
    must_change_from_initial=True,
    min_relevant_steps=4,
    min_state_changes=3,
    min_dependency_depth=4,
)
```

Recommended operation vocabulary:

```text
Put
Move
Remove
```

This gives a relatively clean temporal state-tracking condition.

---

## B. Interleaving

Use:

```python
QuerySpec(
    query_type="location",
    must_change_from_initial=True,
    min_relevant_steps=5,
    min_state_changes=3,
    min_interleaving=0.4,
)
```

The trajectory should contain operations concerning unrelated entities between relevant updates.

Conceptually:

```text
target update
distractor update
distractor update
target update
distractor update
target update
```

This tests whether the model can maintain the target state across interference.

---

## C. Revision

Use:

```python
QuerySpec(
    query_type="location",
    must_change_from_initial=True,
    min_relevant_steps=5,
    min_state_changes=3,
    require_revision=True,
)
```

Revision should represent cases where an earlier interpretation of the state must be overwritten by later information.

For robust dataset construction, explicit revision templates are preferable to relying on random sampling.

---

## D. Redo validity

Use:

```python
force_redo_probe=True
```

This creates the dedicated history condition:

```text
operation
Undo
new operation
```

where the new operation invalidates the redo stack.

---

# 30. Important Experimental Factors

The benchmark is intended to vary several dimensions.

## Entity count `E`

Controls the number of entities the simulator may create.

Higher `E` increases the number of possible objects and therefore potential state complexity.

---

## Update count `U`

Controls the number of canonical state operations.

Higher `U` increases temporal depth.

However:

> `U` alone does not guarantee a difficult query.

This is why `QuerySpec` is needed.

---

## Distractor count `N`

Controls irrelevant textual information.

It changes the input sequence without changing canonical state transitions.

This measures interference from irrelevant context.

---

## Revision/history factor `R`

Tracks the presence of history-related operations such as:

```text
Undo
Redo
```

This introduces a different source of state complexity because the model must track not only the world but also operation history.

---

# 31. A Critical Methodological Point: Factor Independence

`E`, `U`, and `N` should not automatically be treated as perfectly orthogonal.

For example:

```text
higher U
    → longer input
    → more operations
    → potentially more entity interactions
```

Similarly:

```text
higher E
    → potentially more candidate entities
    → potentially more complex state
```

Therefore experiments should record the actual structural properties of each generated instance, not only the requested generation parameters.

Useful metadata includes:

```text
E
U
N
R
relevant_steps
state_change_count
query_change_count
dependency_depth
interleaving
revision
counterfactual sensitivity
```

This allows later analysis to distinguish intended experimental factors from accidental correlations.

---

# 32. Current Successful Example

A generated instance can produce a summary such as:

```text
SUMMARY
{
  "length": 8,
  "world_state_change_count": 8,
  "query_change_count": 2,
  "first_query_change": 0,
  "last_query_change": 4
}
```

The analysis may report:

```text
relevant_steps: [0, 4]
counterfactual_sensitive_steps: [4]
```

and counterfactual probes such as:

```text
remove_step: 4
original_answer: c1
counterfactual_answer: c0
answer_changed: true
```

and:

```text
remove_step: 7
original_answer: c1
counterfactual_answer: c1
answer_changed: false
```

This demonstrates the distinction between:

```text
world-state change
query relevance
causal sensitivity
```

---

# 33. What the Generator Guarantees

For a successfully returned normal instance, the pipeline verifies:

```text
1. Operations are valid.
2. Renderer replay agrees with simulator replay.
3. A query satisfies QuerySpec.
4. The trajectory contains one entry per operation.
5. Step-wise gold has one answer per operation.
6. Final trajectory answer equals direct simulator answer.
7. Serialized gold_answer equals final step-wise answer.
8. Counterfactual probe indices are valid.
9. Counterfactual answers are computed by replay.
```

These checks make the generated dataset auditable.

---

# 34. What the Generator Does Not Guarantee

The generator does not yet guarantee that every requested difficulty factor is statistically independent.

It also does not guarantee that random sampling will efficiently produce rare structural patterns.

For example, asking random sampling to simultaneously produce:

```text
long dependency chain
+
high interleaving
+
revision
+
many entities
```

may have a very low acceptance rate.

The current system handles this through rejection sampling, but that is not the most efficient long-term solution.

---

# 35. Important Limitation: Split

`Split` creates another entity that is identical in type to the source entity.

For example:

```text
The candle split into two identical copies.
```

Now the surface form may contain multiple candles.

A question such as:

```text
Where is the candle now?
```

becomes potentially ambiguous.

The current `NameRegistry` partially handles this by using expressions such as:

```text
one of the candles
```

but identity tracking can still become substantially harder.

Therefore `Split` should be treated as an advanced extension rather than mixed into the simplest location-tracking conditions.

---

# 36. Important Limitation: Merge

`Merge` can move multiple objects simultaneously.

For example:

```text
Everything in the blue closet was moved into the old shelf.
```

This creates broad multi-entity propagation.

It is useful for advanced conditions, but it changes the semantics of "one operation = one object update."

For a clean initial benchmark, it is preferable to establish results using:

```text
Put
Move
Remove
```

before introducing `Merge`.

---

# 37. Important Limitation: Natural-Language Identity

The internal simulator uses stable IDs:

```text
o0
o1
o2
```

but the language layer may refer to them using object types:

```text
the candle
the map
the letter
```

If multiple objects have the same type, surface identity becomes less explicit.

This is a deliberate challenge for entity tracking, but it should be controlled carefully.

A future renderer could introduce controlled identity descriptors if experiments require explicit multi-entity coreference.

---

# 38. Current Generation Philosophy

The revised system should be understood as:

```text
WORLD
  ↓
valid trajectory
  ↓
STRUCTURAL ANALYSIS
  ↓
QuerySpec filtering
  ↓
CANONICAL GOLD
  ↓
LANGUAGE RENDERING
  ↓
TEXTUAL INTERFERENCE
```

not:

```text
language
  ↓
LLM interpretation
  ↓
gold state
```

The LLM is the evaluated system.

It is not part of benchmark-instance generation.

---

# 39. No LLM Paraphrasing

The benchmark deliberately does not use an LLM to paraphrase generated instances.

All surface text is generated by deterministic templates/renderers.

This is important for reproducibility.

If an LLM were used to paraphrase:

```text
A candle was moved from c0 to c1.
```

the generator would introduce another source of variation:

```text
syntactic variation
semantic drift
entity-reference ambiguity
omissions
hallucinated details
```

The current design keeps those factors under explicit programmatic control.

---

# 40. Recommended Next Step

The strongest next architectural improvement is to add explicit trajectory constructors.

Instead of relying entirely on:

```python
sample_sequence()
```

for every condition, introduce deterministic or semi-deterministic constructors such as:

```text
basic_chain
interleaved_chain
cancellation
revision
undo_redo
```

The desired architecture becomes:

```text
                 ┌── basic_chain
                 ├── interleaved_chain
trajectory  ─────┼── cancellation
constructors     ├── revision
                 └── undo_redo
                        ↓
                  structural analysis
                        ↓
                    QuerySpec
                        ↓
                  canonical gold
```

Random sampling can still be used inside each constructor to vary the concrete entities, containers, and exact operations.

The constructor should control the structural pattern.

The random sampler should provide surface diversity.

---

# 41. Proposed Difficulty Families

A useful benchmark hierarchy is:

```text
Level A — Capacity + Temporal Maintenance
    basic sequential chains

Level B — Interference
    relevant updates mixed with irrelevant updates

Level C — Memory Revision
    earlier state becomes obsolete and must be overwritten

Level D — History Operations
    Undo / Redo semantics

Level E — Advanced Composition
    Split / Merge / Swap / multi-object propagation
```

The first three levels should be established before introducing the more complicated operation semantics.

---

# 42. Recommended Separation of Responsibilities

The repository should preserve the following ownership boundaries.

```text
world/
    What is the true state?
    What operations are valid?
    How does each operation transform the state?

generator/sampler.py
    How do we sample valid trajectories?

analysis.py
    Does this trajectory satisfy the requested structural condition?

generator/probes.py
    What query should be asked?
    What counterfactual probes should be generated?

trajectory.py
    What is the gold answer after every step?

render/
    How is the symbolic trace expressed in natural language?

pipeline.py
    How are all components orchestrated into one benchmark record?
```

Keeping these responsibilities separate is important for future experiments.

---

# 43. End-to-End Example

Suppose the sampler creates:

```text
1. Put candle in c0
2. Move candle to c1
3. Put map in c2
4. Move map to c0
5. Move candle to c2
6. Put letter in c1
7. Move letter to c0
8. Move map to c1
```

The renderer might produce:

```text
A candle was placed in the blue closet.
The candle was moved from the blue closet to the old shelf.
A map was placed in the narrow basket.
The map was moved from the narrow basket to the blue closet.
The candle was moved from the old shelf to the narrow basket.
A letter was placed in the old shelf.
The letter was moved from the old shelf to the blue closet.
The map was moved from the blue closet to the old shelf.
```

For:

```python
LocationQuery("candle")
```

the query answer sequence might be:

```text
step 1 → c0
step 2 → c1
step 3 → c1
step 4 → c1
step 5 → c2
step 6 → c2
step 7 → c2
step 8 → c2
```

Therefore:

```text
world-state changes = 8
query-answer changes = 2
```

The final answer is:

```text
c2
```

The important point is that the benchmark measures the model's ability to maintain the relevant state through the entire sequence, rather than merely rewarding it for noticing the final sentence.

---

# 44. Final Design Principle

The central invariant of DWS-Bench is:

> The canonical simulator determines reality; trajectory analysis determines whether an instance is suitable; the renderer determines how that reality is presented to the model.

This gives the benchmark three useful properties:

```text
Deterministic
    same seed → same generated instance

Auditable
    every gold answer can be replayed from operations

Controllable
    structural difficulty can be specified explicitly
```

The revised generator therefore provides a stronger foundation for studying SLM failures in dynamic world-state tracking than unconstrained random narrative generation.
