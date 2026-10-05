# DWS-Bench Generator: Complete Implementation Reference

**Repository state described:** 2026-10-05, commit `447116e`
**Purpose:** explain how the generator currently works, which decisions govern it,
and where the implementation still differs from the written contract.

This is an implementation reference, not a proposal. It distinguishes:

- **Implemented behavior** — what the current Python code actually does.
- **Contract decision** — what `SPEC.md`, `AGENTS.md`, `requirements.md`, or
  `documentation/DECISIONS.md` says the system must do.
- **Open divergence** — a place where those two are not yet the same.

Authority order:

1. thesis proposal (not in this repository);
2. [`SPEC.md`](../SPEC.md);
3. [`AGENTS.md`](../AGENTS.md);
4. [`requirements.md`](../requirements.md) and [`documentation/DECISIONS.md`](DECISIONS.md);
5. docstrings and comments;
6. executable code.

A divergence below is recorded, not silently resolved in favour of code.

---

## 1. Generator purpose

DWS-Bench is a symbolic-first benchmark generator for dynamic world-state tracking.
It builds valid operation trajectories, replays them once, derives gold and
step-wise gold from that replay, renders the trace as natural language, measures the
realized factors, and records enough metadata to audit the instance.

> The simulator is the source of truth. Natural-language text is a rendering of the
> simulator trace, never a second state machine.

The generator isolates: target-relevant update depth (`T`), distractor updates
(`D`), lifetime entity count (`E`), revision complexity (`V`), narrative
distractors (`N`), canonical update count (`U`), rendered length (`L`), and
operation family.

The generator makes no model-performance claim. **The working tree contains no
dataset and no model results**: `data/` and `results/` are gitignored and absent.

---

## 2. Three research questions

| RQ | Question | Experiment |
|---|---|---|
| RQ1 | How does target-relevant depth affect recovery of the final world state, and does degradation depend on mutation structure rather than depth alone? | `experiments/rq1_mutation_depth.py` |
| RQ2 | With the target trajectory fixed, how do state-changing distractors, narrative distractors, and context length differentially affect reasoning, and does superseded information produce a distinct stale-state failure? | `experiments/rq2_interference.py` |
| RQ3 | How do model scale and explicit reasoning prompting alter accuracy and failure profile, and does prompting interact with mutation structure? | `experiments/rq3_scale_reasoning.py` (evaluation only) |

D-018 collapsed the earlier five-RQ structure into these three; RQ4 entity load
became a covariate and the RQ5 structural pilot became RQ1's mutation axis.

---

## 3. Repository layers

```
world            <- render, analysis
world            <- generator
render, analysis <- generator
analysis         <- eval (scoring helpers only, in the offline audit script)
```

`world/` owns `WorldState`, `History`, operation validity and application,
undo/redo semantics, and canonical replay. It knows nothing about prompts, models,
factors, or templates.

`generator/` owns family registries, constructors, family validation, factor
measurement, query and probe construction, the structural-causality gate, and the
single instance gate.

`render/` owns the name vocabulary, operation sentences, distractor sentences, and
question wording. It consumes replay output and never mutates state.

`eval/` is downstream: prompts, model registry, engines, extraction, scoring,
baselines, robustness.

`analysis/` owns query analysis, the first-error taxonomy, curve fits, failure
onset, and the solubility audit.

`torch` and `transformers` may only be imported by `eval/engine.py`. `world` and
`generator` import without them.

---

## 4. World state model

`world/state.py`:

```python
WorldState(
    object_type: Dict[str, str],
    location: Dict[str, str],
    containers: Set[str],
    step_index: int = 0,
)
```

- `object_type` maps every entity that has ever existed to its type and **never
  shrinks**. A removed object keeps its type entry.
- `location` is a partial map of *placed* objects only. A removed object has no
  `location` entry, so a `None` gold means "not placed", not "never existed".
- `containers` is fixed for the trajectory. Operations move objects between
  containers but never create or delete one. Default 3, minimum 2.
- `step_index` advances once per successfully applied operation, including `Undo`
  and `Redo`. It is not `T`.
- `History` holds `undo_stack` and `redo_stack` as **complete cloned states**, not
  inverse operations.

---

## 5. Operation semantics

`world/operations.py` defines the dataclasses and `apply_op`, the single state
transition dispatcher.

| Operation | Validity | Effect |
|---|---|---|
| `Put(obj_id, obj_type, container)` | id never seen; container in `C` | creates type and location entries |
| `Move(obj_id, dst)` | placed; `dst` in `C`; `dst` != current | changes one location entry |
| `Remove(obj_id)` | placed | deletes location entry, keeps type entry |
| `Split(source_obj_id, new_obj_id)` | source placed; new id unseen | child copies type, placed in the source's container |
| `Merge(src_container, dst_container)` | `src != dst`; both in `C`; `src` non-empty | moves all source contents to destination |
| `Swap(container_a, container_b)` | `a != b`; both in `C`, empties allowed | atomic exchange of contents |
| `Undo` | undo stack non-empty | clones current to redo stack, restores previous full state |
| `Redo` | redo stack non-empty | restores the last undone state, clones current to undo stack |

Every non-history operation pushes a clone of the prior state onto `undo_stack` and
**clears `redo_stack`**. That rule is what makes redo-validity probes meaningful.

`Put` is setup. It is excluded from `U`, `T`, and `D`.

`Remove` remains available to the simulator and the sampler but is not emitted by
any released family (D-006 in `SPEC.md` §5, `requirements.md` §4).

---

## 6. Canonical replay

`world/replay.py`:

```python
replay_trace(ops, containers, history=None) -> (trace, final_state, final_history)
```

Each trace element is `(operation, state_before, state_after)`. Replay starts from
an empty world and applies every operation through `apply_op`. One pass feeds:

- final-state gold;
- step-wise gold;
- factor measurement;
- narrative rendering;
- structural-causality counterfactual replays;
- constructor/replay equality checks.

Invalid operations raise `InvalidOperation`. That is deliberately distinct from a
valid replay whose answer is unchanged, which matters for counterfactual deletion.

---

## 7. Family taxonomy

`generator/dataset_spec.py` groups the eight families into five capability groups.

| Capability group | Families | Phenomenon |
|---|---|---|
| Sequential state tracking | `basic_chain`, `revision` | ordinary chains, revisits |
| Multi-entity interference | `interleaved_chain` | target updates mixed with other entities |
| Identity transformation | `split_chain`, `merge_chain` | entity creation, whole-container fusion |
| Global state operations | `swap_chain` | bilateral container exchange |
| Temporal edit history | `undo_chain`, `undo_redo_chain` | rollback and reapplication |

**Open divergence (SPEC OPEN-6).** The family list exists in four places:
`generator/trajectories.py::_CONSTRUCTORS`,
`generator/sampler.py::_CONSTRUCTORS`,
`generator/dataset_spec.py::FAMILY_TO_CAPABILITY_GROUP`, and the validation sets in
`trajectory_specs.py`. They are currently equal, but nothing enforces that.

**Open divergence (SPEC OPEN-7).** `OBJECT_TYPES` is declared twice with different
contents: `render/names.py:12` (the owner per `AGENTS.md` §5) and
`generator/trajectories.py:274`.

---

## 8. Requested condition model

`Condition` in `generator/dataset_spec.py` represents an experimental request:
`family`, `T`, `E`, `D`, `experiment`, `generation_status`.

Derived:

```text
U = T + D
S = initial_placements + U
```

`initial_placements` is normally `E`; `split_chain` places one entity and creates
the second through `Split`.

Numeric checks enforce `T >= 1`, `E >= 1`, `D >= 0`. The old `D < T` restriction was
removed deliberately: `D >= T` is legal (RESOLVED-2).

Generation status lifecycle:

```text
PENDING_CALIBRATION -> PENDING_GENERATION -> GENERATED
```

`GENERATED` is immutable. No condition has reached `GENERATED` in this working
tree, because no dataset exists.

---

## 9. `TrajectorySpec`

`generator/trajectory_specs.py` is the constructor-level specification: `family`,
`entity_count`, `num_containers`, `total_updates`, `target_updates`,
`distractor_updates`, `min_interleaving`, `revision_count`,
`min_unique_target_locations`, `structural_ops`, and an optional `target_obj`.

Generic checks: family is registered, entity count positive, container count meets
the minimum, updates positive, distractor count non-negative, interleaving in
`[0, 1]`, revision count non-negative, minimum unique target locations positive.

For non-structural families it also requires `total_updates == target_updates +
distractor_updates`. For structural families that check is skipped, because the
constructors read `target_updates` under family-specific conventions — one of the
contract ambiguities in `requirements.md` §16.

**Resolved since the last audit.** The transitional `schema_version` property is
gone. `TrajectorySpec` has no `schema_version` field, `generate.py` no longer
advertises `--schema-version`, and the record schema contains no `schema_version`
key. `test/test_factor_contract.py` asserts that constructing with
`schema_version="v2"` fails. The v1/v2 duality described in earlier revisions of this
document is closed.

**Open divergence (SPEC OPEN-15).** `revision_count` is accepted and stored but never
enforced against measured `V`. This is the single test failure not named
`test_failsnow_*`: `test_known_findings.py::test_f10_revision_count_is_enforced`.

---

## 10. Trajectory construction

`generator/trajectories.py` holds the family constructors. Every helper routes
operations through `apply_op`. Each builds an empty world with fixed containers,
adds setup `Put`s, adds family-specific target and distractor operations, and
returns a `ConstructedTrajectory` (`ops`, `containers`, `final_state`, `history`,
`target_obj`, `spec`, optional measured factors).

`build_trajectory` is the only normal entry point: it selects the constructor from
`_CONSTRUCTORS`, replays, compares constructor state against replay state, runs
`validate_trajectory`, and attaches measured factors. Direct constructor calls
happen only in tests (AGENTS.md hard rule 4).

Family shapes:

- **`basic_chain`** — one entity, `D=0`, every post-setup op moves the target to a
  new destination.
- **`interleaved_chain`** — at least two entities, at least one target and one
  distractor move, and genuine interleaving: a distractor must fall between two
  consecutive target moves. Validation computes an interleaving score
  (distractors between target moves ÷ all move ops) against `min_interleaving`.
- **`revision`** — one entity, `D=0`, `T >= 3`, at least `min_unique_target_locations`
  distinct destinations, and a genuine revisit: a location recurring after at least
  one different location in between.
- **`split_chain`** — two entities, `query_type="count"` required, a pre-split move,
  then `Split` then `Merge` with `Merge` after `Split`.
- **`merge_chain`** — at least two entities, `query_type="count"` required, one
  `Merge`, and `Put` count equal to `entity_count`.
- **`swap_chain`** — at least two entities, one `Swap`.
- **`undo_chain`** — one entity, at least one `Move` before the first `Undo`, and
  **no** `Redo`.
- **`undo_redo_chain`** — one entity, at least two `Undo` and one `Redo`, at least
  three `Move`s before the first `Undo`, and the trace must end with `Redo` so the
  final answer depends on it.

### 10.1 Enforced family minima vs contract

`T_min` in the `SPEC.md` frontmatter is the requested **target-update** floor
(`TrajectorySpec.target_updates`, symbol $T$), not a total-op count. These are the
floors `generator/trajectory_validation.py` actually rejects below:

| Family | Enforced | `SPEC.md` frontmatter | `requirements.md` §4 |
|---|---:|---:|---:|
| `basic_chain` | `T >= 1` (from `TrajectorySpec`) | 1 | 1 |
| `revision` | `T >= 3` | 3 | 3 |
| `interleaved_chain` | `T >= 1`, `D >= 1`, `E >= 2` | 1 | 1 |
| `split_chain` | **`T >= 3`** | 2 | composition only |
| `merge_chain` | **`T >= 1`** | 2 | composition only |
| `swap_chain` | `T >= 1` | 1 | 1 |
| `undo_chain` | `T >= 2` | 2 | 2 |
| `undo_redo_chain` | **`T >= 6`** | 3 | composition only |

Three divergences, tracked as SPEC OPEN-17. The contract wins; the validator must be
corrected. In practice the enforced minimum is stricter, so a legal-per-contract
request is rejected rather than silently accepted — a safe direction, but still a
contract violation. The global `T >= 1` floor lives in `TrajectorySpec.__post_init__`
(`generator/trajectory_specs.py:128`), not in the validator.

`SPEC.md` also declares `containers_min: 2` for `swap_chain` while `requirements.md`
§4 asks for at least three. Only two are enforced, by `TrajectorySpec.__post_init__`
against `NUM_CONTAINERS_MIN` and by `_initial_world`
(`generator/trajectories.py:81`). RQ1 requests three, so released `swap_chain`
records satisfy both.

---

## 11. Family validation

`generator/trajectory_validation.py` runs after construction.

**Generic (non-structural) checks:** non-empty trace; target created by a `Put`;
exactly `entity_count` `Put` operations; no duplicate creation; correct
target/distractor counts.

**`interleaved_chain`:** `E >= 2`, `T >= 1`, `D >= 1`, at least one target and one
distractor `Move`, real interleaving, and `min_interleaving`.

**`revision`:** `E == 1`, `D == 0`, `T >= 3`, enough unique destinations, genuine
revisit.

**Structural families:** `split_chain` (`E == 2`, `T >= 3`, `query_type="count"`,
a `Split`, a `Merge`, `Merge` after `Split`); `merge_chain` (`E >= 2`, `T >= 1`,
`query_type="count"`, a `Merge`, `Put` count matches); `swap_chain` (`E >= 2`,
`T >= 1`, a `Swap`); `undo_chain` (`E == 1`, `T >= 2`, an `Undo`, no `Redo`, a
pre-`Undo` `Move`); `undo_redo_chain` (`E == 1`, `T >= 6`, two `Undo`s, a `Redo`
after the first `Undo`, three pre-`Undo` `Move`s, last op is `Redo`).

A family with no validator raises `ValueError`, so an unregistered family cannot
pass silently.

---

## 12. Structural causality

`generator/structural.py::validate_structural_causality` is the single
implementation. It is called from `build_trajectory` for every structural family
and is not duplicated in the constructors.

Declared requirements, `generator/dataset_spec.py::REQUIRED_STRUCTURAL_OPS`:

| Family | Required operation |
|---|---|
| `split_chain` | `Split` |
| `merge_chain` | `Merge` |
| `swap_chain` | `Swap` |
| `undo_chain` | `Undo` |
| `undo_redo_chain` | `Undo`, `Redo` |

Checks, in order:

1. **presence** — every required operation type appears;
2. **target effect** — at least one occurrence of each required type changes the
   target. This is read off the replay (`state_before.location[target] !=
   state_after.location[target]`), with a `Split` special case for the target being
   the source or the spawned child. A field-matching predicate is explicitly
   rejected here, because `Undo`/`Redo` carry no reference to the event they
   reverse;
3. **necessity** — deleting **each** occurrence either makes replay invalid or
   changes the gold answer. For `query_type="count"` the compared quantity is the
   count of the target's type in the `Merge` destination, not the target's location;
4. **no trailing ordinary move** — a trailing target `Move` must not be the final
   determinant. Exempt families: `split_chain` (the `Split` creates its child in the
   source's container, so a later `Move` of that child is part of the intended
   chain) and `undo_redo_chain` (history operations are intentionally placed before
   final moves).

Check 5 runs after the counterfactual replays on purpose: `Move` writes an absolute
location, so a trailing target `Move` also trips check 3 and would mask the real
cause.

**Open divergence (SPEC OPEN-18) — live defect.** Lines 165–196 are a second copy of
the count/location comparison that was left *outside* the `for found_idx` loop that
binds `counterfactual`. When every occurrence deleted at that step raised
`InvalidOperation`, the loop body never executes, `counterfactual` is unbound, and
Python raises `UnboundLocalError` instead of a structural-causality verdict. This
is what breaks `split_chain` in `test/test_invariants.py` and, indirectly, several
`test_failsnow_*` markers. It is a code bug, not documentation drift.

Measured reach, 15 seeds per cell, 50 retry attempts each, dedup registry reset
between cells:

| Condition | Outcome |
|---|---|
| `split_chain T=4 D=0` | 14 accepted, 1 rejected |
| `split_chain T=8 D=0` | 15 accepted |
| `split_chain T=16 D=0` | 15 accepted |
| `split_chain T=4/8/16 D=1` | 0 accepted, `structural_causality` |
| `split_chain T=4/8/16 D=2` | 0 accepted, `structural_causality` |

So the trigger is trace shape, not the family name: with at least one distractor
update the sampler emits a single `Split`, every deletion is invalid, and the cell
becomes unreachable. RQ1 asks for `D=0`, so the released grid mostly escapes it, but
`test_invariants.py` sweeps `D=1` and dies.

---

## 13. Factor measurement

`generator/metadata.py` is the single owner. `MeasuredFactors` carries
`E_actual`, `T_actual`, `D_actual`, `V_actual`, `N_actual`, `L_word`.

- **`E`** — unique ids created by `Put.obj_id` and `Split.new_obj_id`. The
  implementation also folds in `Split.source_obj_id`, which cannot change the union
  because the source must already exist.
- **`T` / `D`** — state-based. For every post-setup operation, compare the target's
  location before and after; changed means `T`, unchanged means `D`. This is
  deliberately not an operation whitelist, so `Split`, `Merge`, `Swap`, target-
  affecting `Undo`/`Redo`, and `Remove` are classified by their actual effect.
  `classify_op` carries the exhaustive boundary table (target as split source vs
  child, target in merge source/destination/neither, target on either side of a
  swap, and so on).
- **`U`** — `T + D`; every non-`Put` operation lands in exactly one bucket.
- **`V`** — target-location revisits plus target-affecting `Undo`/`Redo`. A revisit
  requires returning to a previous destination after an intervening different one.
  **Open divergence (SPEC OPEN-4):** a target-affecting `Undo` can be counted both
  as a revisit and as a reversal. The double count is pinned by tests and still
  undecided.
- **`N`** — pure-text distractor sentences with no symbolic transition. Measured
  from the sentence list when sentences are supplied, otherwise from the explicit
  requested count.
- **`L_word`** — `sum(len(sentence.split()) for sentence in sentences)`. The
  generation ceiling is `L_MAX_WORDS = 600`. Tokenizer length `L_tok` is a
  downstream diagnostic recorded by `eval/engine.py` and never a gate.

`L_actual` no longer exists. It duplicated `L_word` under a second name and was
removed; only `L_word` is serialized.

`verify_factors` asserts exact equality for `E`, `T`, `D`, and checks `V` only when
the caller supplies `min_v` or `intended_v`. `verify_length` asserts the word
ceiling, so it must be called after rendering — an unrendered sentinel length of
`-1` would pass.

---

## 14. Query construction

`generator/probes.py` defines `LocationQuery`, `CountQuery`, and
`RedoValidityQuery`.

- Location gold is `state.location.get(obj_id)`, so a removed object's gold is
  `None`.
- Count gold is the number of objects of a type currently in a container.
- Redo-validity gold reads history (`can_redo(history)`), not `WorldState`, which is
  why records must preserve history or a derived label.

`analysis/query_analysis.py` can filter candidate queries on measured properties:
relevant steps, dependency depth, interleaving, revisions, answer change.

**Open divergence (SPEC OPEN-5).** There are two notions of "target-relevant":
state-change in `metadata.py` (used for `T`/`D`) and syntactic in
`query_analysis.py` (used for candidate filtering). They can disagree. Both are
carried; neither is authoritative for the other.

---

## 15. Counterfactual probes

Every candidate removal is classified into exactly one bucket:

1. `valid_answer_changing`
2. `valid_answer_preserving`
3. `invalid_replay`
4. `excluded_setup`

For each candidate: remove exactly one operation, replay, classify invalid replay
separately from a valid replay with an unchanged answer, and record the index and
outcome. Setup `Put` operations are excluded by default, because deleting setup
changes initial conditions rather than intervening on an event.

The accounting object asserts that the four buckets conserve all candidates. A
valid `None` answer (target removed) is distinct from an invalid replay. Probe
selection prefers one sensitive plus one insensitive valid deletion and reports the
degeneracy rather than presenting it as balanced.

Redo-validity probes are built in two classes: valid redo (`Undo` then stop) and
invalid redo (`Undo`, then a new ordinary operation that clears the redo stack). The
label is read from history, never assumed from the requested class.

---

## 16. Rendering

`render/narrative.py` is the only renderer. `render/templates.py` has been deleted,
so the duplicate-renderer risk described in earlier revisions of this document is
closed.

`RENDER_DISPATCH` maps each operation class to a sentence renderer. For each
operation the renderer uses the corresponding `state_before` for source-sensitive
wording, then returns the sentence list and final state.

`Move` uses destination-only wording ("The object was moved to container B") for
all families, with no schema-dependent or family-specific transition branch.

`render/names.py` provides the seeded `NameRegistry`, the `OBJECT_TYPES`
vocabulary, container display names, ordinal and duplicate surface-name forms, and
the two distractor helpers:

- `make_distractor_sentences` builds pure-text filler sentences;
- `splice_distractors` inserts them at seeded positions.

Text distractors increase `N` and `L_word` and never touch symbolic state or gold.

---

## 17. The single instance gate

`generator/instance.py` is the one request-to-record path. It replaced two
producers that had disagreed on the key set and had never called
`verify_factors` or `verify_length`.

`build_validated_instance` runs, in order:

| Step | Check | Failure reason |
|---|---|---|
| 1 | `build_trajectory` | `build_trajectory` |
| 2 | render narrative, splice distractors | `render_narrative` |
| 3 | `measure_factors` | `measure_factors` |
| 4 | structural causality | `structural_causality` |
| 4b | `verify_factors` | `verify_factors` |
| 4c | `verify_length` | `verify_length` |
| 4d | answer leakage | `answer_leakage` |
| 4e | duplicate trace | `duplicate_trace` |
| 5 | record assembly and schema assertion | — |

A rejection returns `InstanceGateFailure` naming the check, the requested and
measured values, and a description. It is never swallowed into a bare `ValueError`.

`generate_instance_with_retry` counts rejection reasons across attempts and raises
`GenerationError` with the full histogram plus the first three failures once the
budget (default 50) is spent. Attempt 0 uses the caller's seed verbatim; later
attempts use `attempt_seed(seed, attempt)`, a SHA-1-derived sub-seed, so a retry is a
fresh trajectory rather than a rerun of the same draw.

### 17.1 Record schema

`INSTANCE_RECORD_KEYS` declares the schema once, and `build_validated_instance`
asserts the assembled record matches it exactly, in order. A field added on one side
only fails loudly with `raise AssertionError` so the two cannot drift.

```text
instance_id, family, experiment, condition_id, seed, trace_hash, attempt,
generator_version, renderer_version, scoring_version,
requested_factors, measured_factors, spec, canonical_trace, sentences, context,
question, query_entity, gold_container, gold_answer,
step_wise_gold, step_wise_gold_answers, final_state
```

There is no `schema_version` and no `query_type` field. The `spec` sub-dict carries
`entity_count`, `target_updates`, `distractor_updates`, `num_containers`,
`total_updates`, `initial_placements`, `total_transitions`.

**Open divergence.** `query_type` is not serialized, so a consumer cannot tell from
the record whether the question is a location or a count question.
`test_questions_and_records.py::test_failsnow_record_serialises_the_query_type` marks
this.

`trace_hash` is SHA-1 over the serialized canonical trace only. It does not cover
family, target identity, or container set, and the `_SEEN_TRACE_HASHES` registry is
process-global, so duplicates cannot be detected across separate generation
processes. See SPEC RESOLVED-13 and `requirements.md` §12.

### 17.2 Count-query gold

Count-query handling exists **only for `split_chain`**:

```python
if spec.family == "split_chain" and spec.query_type == "count":
    merge_dst = first Merge operation's destination
    question = question_count(merge_dst, target_type, names)
    gold_answer = str(count)
```

Everything else uses `question_location` and a container display name. This matters
because `experiments/rq1_mutation_depth.py::FAMILY_QUERY_TYPES` requests
`query_type="count"` for `merge_chain`, `swap_chain`, and `undo_chain` too. Those
records therefore carry a location question while their family contract says count.
`gold_container` is always the target's final location, never the `Merge`
destination the count question asks about, and `query_type` is not serialized, so
downstream analysis cannot detect the substitution.

Verified on 2026-10-05 through `experiments/_common.generate_instance`, requesting
`query_type="count"`:

| Family | Result |
|---|---|
| `swap_chain T=4 D=0` | accepted; `question='Where is the coin now?'`, `gold='the tall basket'` |
| `undo_chain T=4 D=0` | accepted; `question='Where is the map now?'`, `gold='the old chest'` |
| `merge_chain T=4/8 D=0` | never accepted; `verify_factors:T` on all 50 attempts, every seed |

`merge_chain` is separately unreachable: the sampler never reproduces the requested
$T$, so RQ1 would emit zero `merge_chain` records. The dry-run probe does not catch
this, because `probe_reachability` calls `build_trajectory` and never enters the gate.

This is the root of the red `test_questions_and_records.py` markers:

- `test_failsnow_count_query_renders_a_count_question_for_every_count_family`
- `test_failsnow_gold_container_is_the_container_the_question_asks_about[split_chain-T4]`
- `test_failsnow_step_wise_gold_answers_are_counts_for_count_cells`

### 17.3 Answer leakage

`_check_answer_leakage(sentences, gold_answer, num_op_sentences, suffix_k=3)` slices
off the operation sentences and checks the last `suffix_k` **distractor** sentences
for the lower-cased gold as a substring. The caller passes `distractors` directly
with `num_op_sentences=0`, and the check only runs when
`textual_distractor_count` is non-zero.

Consequences, tracked as SPEC PARTIAL-10:

- a record with `N = 0` is never checked, and all RQ1 conditions have `N = 0`;
- `suffix_k` is a keyword default rather than a `SPEC.md` constant;
- normalization is substring matching after `lower().strip()`, not punctuation
  trimming and article removal;
- no rejection reason or location is persisted.

---

## 18. Experiment generation pipeline

`experiments/_common.py` is the batch path: `generate_instance` derives a seed and
calls `generate_instance_with_retry`; `generate_condition` loops a cell and returns
records plus a failure count; `verify_generated_records` re-checks a written file;
`print_factor_summary` prints the measured distributions.

`verify_generated_records` is a post-hoc re-read of the JSONL, not a second
generator.

### 18.1 RQ1 — mutation and depth

`experiments/rq1_mutation_depth.py`, `D = 0`, `N = 0`, 50 instances per condition:

| Family | E | T levels | Query type | Conditions |
|---|---:|---|---|---:|
| `basic_chain` | 1 | 2, 4, 6, 8, 12, 16 | location | 6 |
| `revision` | 1 | 4, 8, 12, 16 | location | 4 |
| `split_chain` | 2 | 4, 8, 12, 16 | count | 4 |
| `merge_chain` | 2 | 4, 8, 12, 16 | count | 4 |
| `swap_chain` | 2 | 4, 8, 12, 16 | count | 4 |
| `undo_chain` | 1 | 4, 8, 12, 16 | count | 4 |
| `undo_redo_chain` | 1 | **6**, 8, 12, 16 | location | 4 |

30 conditions × 50 = **1,500 records**. Every family uses 3 containers.
`undo_redo_chain` starts at 6 because the validator requires `T >= 6`.

### 18.2 RQ2 — interference and supersession

`experiments/rq2_interference.py`: `interleaved_chain`, E=3, T=8, 4 containers,
`LENGTH_MATCHED = True` (a flag with no implementation behind it).

| Condition | D | N | Conditions |
|---|---:|---:|---:|
| `D4`, `D8`, `D16` | 4, 8, 16 | 0 | 3 |
| `D4_N4`, `D4_N8`, `D4_N16` | 4 | 4, 8, 16 | 3 |
| `revision_T8` supersession | 0 | 0 | 1 |

7 conditions × 50 = **350 records**. `D = 0` is not a cell; the
`basic_chain T=8` record is treated as the baseline.

**Open divergence.** `generate_all.py::EXPERIMENT_SCRIPTS` declares an expected RQ2
count of 700, and the module docstring repeats it. The tuple element is unpacked as
`expected_count` and then never read, so nothing validates or reports it. The RQ2
script defaults to 50 instances per condition and therefore plans 350. The stale
700 predates the current default and must be corrected.

**Open divergence.** The `N` sweep is not length matched. See
[`documentation/LENGTH_CONTROLS.md`](LENGTH_CONTROLS.md).

### 18.3 RQ3 — scale and reasoning

`experiments/rq3_scale_reasoning.py` is evaluation-only: 0.5B / 3B / 7B × direct /
structured over representative RQ1 and RQ2 conditions. It generates no data.
`test_failsnow_rq3_condition_ids_are_unique_per_family` is red, so its condition ids
are not yet unique per family.

---

## 19. Lightweight CLI path

`generate.py` produces family examples through `build_validated_instance`. It
exposes `--family`, `--count`, `--seed`, `--output`. It has no `--schema-version`
flag any more.

Because it now goes through the shared gate, the earlier finding that it skipped
`verify_factors`/`verify_length` is closed. It is still not a substitute for the
experiment pipeline: it is not an RQ sweep and it does not enforce a grid.

---

## 20. Determinism

Required and implemented:

- explicit `random.Random` instances only, no module-global `random.*` calls;
- no time, uuid, or ambient process state in instance content;
- no set-iteration order in serialized content;
- stable serialization and a per-record `trace_hash`;
- separate trajectory and naming RNG streams, so changing surface names does not
  shift symbolic construction.

Intended property:

```text
same condition + same seed + same code/spec -> identical trace and identical bytes
```

`test_experiment_scripts.py::test_rerun_regenerates_a_byte_identical_file` covers
the rerun case.

Changing RNG consumption, wording, gold extraction, or factor semantics requires a
SPEC version bump and regeneration.

**Open divergence (SPEC OPEN-16).** `SPEC.md` declares `version: 0.3.0` while
`generator/constants.py::SPEC_VERSION` is `"0.2.0-v2"` and is what
`generator_version`, `renderer_version`, and `scoring_version` stamp into every
record. Data provenance and contract version therefore disagree.

---

## 21. Release gates and what is missing

Before a condition may move to `PENDING_GENERATION`, the process requires: contract
consistency, import smoke, lint/type, unit and integration tests, a seed sweep, a
family reachability sweep, factor verification, a structural-causality audit, probe
accounting, a leakage audit, a solubility pilot, trace deduplication, a length
audit, and manual pilot inspection.

Implemented and runnable:

- reachability probe — `generate_all.py --dry-run`, currently green for RQ1 and RQ2;
- the ordered instance gate, including factors, length, leakage, and dedup;
- byte-identical rerun test.

Missing, and blocking a `GENERATED` declaration:

1. a release manifest builder and a manifest file next to the data (three
   `test_failsnow_*` markers in `test_reproducibility_freeze.py`);
2. `revision_count` enforcement (SPEC OPEN-15);
3. the `generator/structural.py` defect (SPEC OPEN-18);
4. version-stamp reconciliation (SPEC OPEN-16);
5. family-minimum reconciliation (SPEC OPEN-17);
6. count-query gold for `merge_chain`, `swap_chain`, and `undo_chain` (§17.2);
7. a leakage check that inspects operation sentences, not only distractor tails
   (SPEC PARTIAL-10);
8. an independent factor/gold checker (SPEC OPEN-12);
9. cross-process deduplication;
10. length matching with `matched_*` metadata.

D-001 paused freezing until the factor contract is locked and a pilot is inspected by
hand. D-010 removed the old data and results and forbids treating them as evidence.

---

## 22. Evaluation and analysis

### 22.1 Model registry

`eval/models.py::ModelConfig` is frozen with `hf_model_id`, a pinned commit-hash
`revision`, and decoding fields `temperature=0.0`, `top_p=1.0`, `do_sample=False`,
`max_new_tokens=256`. Core: `qwen2.5-0.5b`, `qwen2.5-3b`, `qwen2.5-7b`,
`llama-3.2-3b`, `olmo-2-1b`. Optional: `phi-4-mini`, `olmo-2-7b`.

`HuggingFaceEngine.generate_batch` defaults to the model config's
`max_new_tokens` and enforces greedy decoding unless `enforce_greedy=False`. This
closes SPEC OPEN-8 (RESOLVED-8).

### 22.2 Scoring

`eval/scoring.py` is the single scoring authority. It takes the first `Final Answer:`
line so echoed continuation text cannot replace the answer, normalises candidates
(lower-case, trimmed punctuation, leading article dropped), and requires an exact
single-candidate match. `strict_correct = semantic_correct AND
protocol_compliant`; under CoT, protocol requires `Step k:` lines. Candidates are
step-wise gold ∪ gold ∪ final-state container display names.

Count gold is a decimal string (`gold_answer = str(count)`), so scoring only ever
sees strings (RESOLVED-9).

`eval/eval_harness.py` reports `format_compliance_rate` per condition and
`overall_format_compliance_rate` overall, separating format failure from reasoning
failure. One gap remains red:
`test_failsnow_wrong_numeric_answers_are_extracted_as_wrong_not_dropped`.

### 22.3 Not wired into `run_eval.py`

Three modules are implemented and tested but never invoked by the CLI:

| Module | Provides |
|---|---|
| `eval/baselines.py` | `compute_stateless_baseline`, `compute_mfc_baseline`, `summarize_baselines`, `run_all_baselines`, `BaselineResult` |
| `eval/robustness.py` | `PROMPT_TEMPLATES` (3 variants), `build_prompt_variants`, `apply_paraphrase`, `evaluate_prompt_sensitivity`, `evaluate_paraphrase_robustness`, `run_robustness_suite` |
| `analysis/solubility.py` | `check_answer_uniqueness`, `check_solubility_llm`, `run_solubility_audit`, `SolubilityResult` |

`run_eval.py` has no `--baselines` or `--robustness` flag, and `--prompt-version`
accepts only `v1` and `v2`, so the third robustness template is unreachable from the
CLI. Dataset shortcuts are `full`, `rq1`, `rq2`; the help text also advertises `rq3`,
which has no dataset entry.

`analysis/evaluate_existing_predictions.py` re-scores prediction files offline
without inference. It does not independently recompute gold, so it does not satisfy
SPEC OPEN-12.

### 22.4 Analysis

`analysis/query_analysis.py` (`QuerySpec`, `analyze_trajectory`, `QueryAnalysis`),
`analysis/first_error.py` (`ErrorType`, `analyze_first_error` with `NO_ERROR`,
`LOCAL_ERROR`, `PROPAGATING_ERROR`, `FINAL_ONLY_ERROR`, `CANCELLATION_ERROR`),
`analysis/failure_onset.py` (`compute_failure_onset` against
`FAILURE_THRESHOLD_TAU = 0.70`), and the curve fits in `analysis/curves.py`
(`fit_linear`, `fit_exponential`, `fit_sigmoid`, `compare_curves`,
`best_fitting_curve`).

All of `test_analysis_curves.py`'s seven tests are `test_failsnow_*`: confidence
intervals, small-noise model selection, parameter recovery, the chance floor, the
50-success cell threshold, and McNemar availability are not yet implemented.

---

## 23. Tests

`test/` is pytest-based: 611 collected, 529 passed, 79 failed, 3 skipped. 78 failures
are `test_failsnow_*` markers for known gaps; the one unmarked failure is
`test_known_findings.py::test_f10_revision_count_is_enforced`.

The known-gap markers cluster as follows.

| Area | Failing markers | Theme |
|---|---|---|
| `test_shortcut_audits.py` | 19 | solvers beat chance; `gold_container` balance; majority-class shortcut |
| `test_baseline_solvers.py` | 15 | stateless and MFC solver ceilings, empty-prediction rate |
| `test_questions_and_records.py` | 10 | count-question rendering, `gold_container`, `query_type` serialization |
| `test_experiment_scripts.py` | 7 | script CLI parity, full-run generation, RQ2 length matching, RQ3 condition ids |
| `test_analysis_curves.py` | 7 | curve-fit statistics and confidence intervals |
| `test_harness_inference.py` | 5 | cell pooling, greedy path, prediction trajectory filling |
| `test_generator_invariants.py` | 5 | RQ1 grid cell generation, query-type map vs builders, `V <= T` |
| `test_rendering.py` | 4 | paraphrase coverage, distractor placement around `Undo`, ordinal order, "original and duplicate" explanation |
| `test_reproducibility_freeze.py` | 3 | manifest builder and manifest file |
| `test_world_sim.py` | 1 | invalid operation names the offending index |
| `test_scoring_extraction.py` | 1 | wrong numeric answers scored wrong, not dropped |
| `test_post_run_sanity.py` | 1 | sanity checks exposed as a helper |

### 23.1 Legacy standalone runner

`python3 test/run_all.py` runs seven scripts directly: `smoke_test`,
`smoke_test_trajectories`, `test_invariants`, `test_measured_factors`,
`test_analysis_and_eval`, `test_eval_pipeline`, `test_scoring`. Five pass, two fail
for code reasons:

- `smoke_test_trajectories.py` constructs a `split_chain` spec without
  `query_type="count"`, which `build_split_chain` now requires, and dies at
  `generator/trajectories.py:749`;
- `test_invariants.py` hits the `UnboundLocalError` from SPEC OPEN-18.

Use pytest as the source of truth.

### 23.2 Audit scripts

`analysis/audits/` holds nine standalone behavioural audit scripts, documented in
its own README. They print summary statistics rather than pass/fail and are not in
CI.

---

## 24. Decisions of record

Summarised from [`documentation/DECISIONS.md`](DECISIONS.md); the log remains the
normative narrative.

| Id | Decision |
|---|---|
| D-001 | Freeze paused until the factor contract is locked and a pilot is inspected |
| D-002 | Two length quantities: `L_word` gates generation, `L_tok` is diagnostic |
| D-003 | One contract; no v1/v2 branching |
| D-004 | Structural causality is declared per family, not inferred |
| D-005 | `generator/metadata.py` owns factor definitions |
| D-006 | `Remove` stays outside released families; `L_actual` alias removed |
| D-007 | Probe accounting conserves all eligible deletions |
| D-008 | Legacy root modules removed |
| D-009 | No family or factor is dropped to make generation pass |
| D-010 | Data and results are regenerated, never fabricated as fixtures |
| D-011 | Thresholds and version stamps live in `generator/constants.py` |
| D-018 | Three-RQ structure; RQ4 and RQ5 folded into RQ1/RQ2 |

---

## 25. Open issues before freezing

1. Fix `generator/structural.py:165-196` (SPEC OPEN-18).
2. Make `merge_chain` reachable, and make `probe_reachability` enter the gate so the
   next unreachable condition is caught before a grid is frozen (OPEN-20).
3. Enforce or retire `TrajectorySpec.revision_count` (OPEN-15).
4. Reconcile `SPEC_VERSION` with the `SPEC.md` frontmatter version (OPEN-16).
5. Reconcile the three family minima (OPEN-17).
6. Implement count-query gold for `merge_chain`, `swap_chain`, `undo_chain`, and
   align `gold_container` with the container the question asks about (OPEN-19, §17.2).
7. Serialize `query_type` on the record (OPEN-19).
8. Extend the leakage gate to operation sentences and lock `k` in `SPEC.md`
   (PARTIAL-10).
9. Decide the `V` double count (OPEN-4).
10. Reconcile state-change relevance with syntactic query analysis (OPEN-5).
11. Centralize the family registry (OPEN-6).
12. Centralize the object-type vocabulary (OPEN-7).
13. Add a release manifest builder and file.
14. Make deduplication cross-process.
15. Implement RQ2 length matching with `matched_*` metadata.
16. Wire baselines, robustness, and solubility into the evaluation path.
17. Add per-condition `L_tok` summaries.
18. Lock an independent factor/gold checker (OPEN-12).
19. Run a seed sweep and inspect a generated pilot by hand.

---

## 26. End-to-end summary

```text
Condition
  -> TrajectorySpec
  -> family constructor (every op through apply_op)
  -> canonical replay trace
  -> constructor/replay equality
  -> validate_trajectory
  -> structural causality (counterfactual replays)
  -> measure_factors
  -> verify_factors / verify_length
  -> answer-leakage check
  -> duplicate-trace check
  -> record assembly with schema assertion
  -> JSONL
```

The engineering invariant: no stage may invent symbolic state. Any sentence, answer,
factor, or probe that cannot be traced back to the canonical replay is not compliant
with the contract.