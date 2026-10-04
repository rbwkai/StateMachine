# DWS-Bench Decision Log

Decisions of record for the DWS-Bench generator, in the order they were taken.

This file is the narrative companion to `SPEC.md`. `SPEC.md` is the machine-readable
contract and wins on any disagreement (AGENTS.md §2 authority order: thesis proposal >
`SPEC.md` > `AGENTS.md` > docstrings > code). Where a decision here changes the contract,
`SPEC.md` is updated in the same change. Where a decision here only settles an open item
that `SPEC.md` already records, the `SPEC.md` §8 row moves from OPEN to RESOLVED.

Format: one entry per decision, each stating what was decided, why, what it replaces, and
what it forces. Entries are immutable once written; a reversal is a new entry that cites
the entry it reverses.

---

## D-001 — Dataset generation paused pending a locked factor contract

**Date:** 2026-10-04
**Status:** in force
**Cites:** user instruction, 2026-10-04

No dataset is to be frozen until the v2 factor contract is unambiguous and a small pilot
has been inspected by hand. The previously committed benchmark and all run outputs under
`results/` are treated as unnecessary and are not to be reused as evidence.

**Forces:** `data/` and `results/` were removed from version control and gitignored; the
full benchmark is regenerated from scratch once the contract is locked.

---

## D-002 — Length is measured in two distinct quantities

**Date:** 2026-10-04
**Status:** in force
**Resolves:** `SPEC.md` §8 RESOLVED-1, restated to remove an ambiguity
**Cites:** user instruction, 2026-10-04

Two quantities exist and must never be conflated:

- $L_{word}$ — rendered narrative **word count**. This is the only length quantity that
  determines generation eligibility.
- $L_{tok}$ — **tokenizer-measured** sequence length. This is reported only as an
  evaluation-side diagnostic. It does not determine gate eligibility.

Where earlier drafts of the plan said "$L$ is tokenizer-measured", that was wrong and is
superseded by this entry.

**Rationale:** a tokenizer inside the generation gate would require `transformers` in
`generator/` or `world/`, which AGENTS.md §4 forbids. Keeping the gate on words preserves
the import-DAG invariant and satisfies SPEC RESOLVED-1.

**Forces:** `generator/constants.py::L_MAX_WORDS = 600` is expressed in words.
`generator/metadata.py::verify_length` is the first code enforcement of the word ceiling;
before this change nothing enforced it anywhere.

---

## D-003 — One contract; the v1/v2 duality is removed

**Date:** 2026-10-04
**Status:** in force
**Supersedes:** the `schema_versions: [v1, v2]` entry in the `SPEC.md` frontmatter and the
`schema_version` field on `TrajectorySpec`
**Cites:** user decision, 2026-10-04

The repository has a single contract, not two schemas. Every `schema_version` branch is
removed. Specifically:

- `TrajectorySpec.schema_version` no longer exists as a constructor field.
- The renderer emits **destination-only** Move wording for every family:
  `"X was moved to B."` The source container is never named in a Move sentence.
- The scorer takes the **first** `Final Answer:` marker in a response, for every record.
  The historical last-marker behaviour is gone.

**Rationale for destination-only wording everywhere:** `SPEC.md`'s v2 structural-causality
rule required it for structural families to stop the source container exposing a shortcut.
Rather than carry two rendering paths, the shortcut class is removed from the whole
benchmark. The source of a Move is recoverable from the preceding sentence anyway, so no
question becomes unanswerable.

**Rationale for first-marker everywhere:** a response that echoes or continues past its
answer would otherwise have its answer replaced by the echo. Under a single contract the
stricter rule applies to everything.

**Consequence:** all previously generated records are invalid, because every rendered
sentence and every scored answer changes. This is consistent with D-001.

**Forces:** `SPEC.md` frontmatter `schema_versions` must be deleted. `SPEC.md`'s
"v1 data and rendering remain legacy-compatible and are not rewritten in place" clause is
deleted.

---

## D-004 — Structural causality is declared per family, not inferred

**Date:** 2026-10-04
**Status:** in force
**Cites:** user instruction, 2026-10-04

The earlier prototype required that **every** structural operation in a trajectory be
causally necessary for the final answer. That is too strong: a trajectory may contain
several structural operations, or structural operations that are distractors with respect
to the queried target.

The rule is now: each structural family declares which operation types are **required**.
For a trajectory of that family, `build_trajectory` must verify that

1. a required structural operation is present;
2. it affects the queried target, either directly or through the target's identity — for
   `split_chain` the queried target is the spawned child, created by the `Split`;
3. removing **that** operation either changes the gold answer or makes the replay invalid;
4. removing an *unrelated* structural operation is **not** required to change the answer;
5. the answer is not determined solely by a final ordinary target `Move`.

Per-family declarations:

| Family | Required ops | Shape |
|---|---|---|
| `split_chain` | `split` | query the spawned child; the `Split` creates the queried target |
| `merge_chain` | `merge` | target sits in the source container; the `Merge` is the last target-affecting event |
| `swap_chain` | `swap` | target sits in one participant; the `Swap` is the last target-affecting event |
| `undo_chain` | `undo` | ends with `Undo`; the undone `Move` would otherwise give a different final location |
| `undo_redo_chain` | `undo`, `redo` | ends with `Undo` then `Redo`; both history transitions validated |

This logic lives in a reusable helper, not inline in `build_trajectory`.

---

## D-005 — Factor definitions have exactly one owner

**Date:** 2026-10-04
**Status:** in force
**Cites:** AGENTS.md §5, SPEC §2, user instruction 2026-10-04

`generator/metadata.py` is the single authority for what $E, T, D, U, V, N, L_{word}$ mean.
Every value is derived from one canonical `replay_trace` pass. Builders, validation, and
record generation all consume it; none of them re-derives a count.

- $E$ — unique entity ids ever created (`Put.obj_id` union `Split.new_obj_id`).
- $T$ — post-`Put` operations that change the queried target's **state or location**. A
  structural operation that changes the target counts in $T$. A `Remove` of the target
  counts, because the location becomes `None`.
- $D$ — post-`Put` operations that do not affect the target.
- $U = T + D$ — all post-initialization operations. `Put` is setup and is excluded.
- $V$ — target-location revisits plus target-affecting history reversals.
- $N$ — pure-text distractor sentences, contributing zero state transitions.
- $L_{word}$ — rendered narrative word count. See D-002.

The T/D discriminator is **state-based, not syntactic**: an operation counts toward $T$
if and only if the replay changed the target's entry in `location`. This lets `Split`,
`Merge`, `Swap`, `Undo`, `Redo` and `Remove` be counted without a per-class whitelist.

**Gate:** `verify_factors` requires measured $E, T, D$ to equal the request **exactly**.
$V$ is a measured, qualified property and is **not** gated on equality unless the caller
explicitly passes `min_v` or `intended_v`. The previous hardcoded
`family == "revision"` special case is gone.

---

## D-006 — `Remove` stays out of the trajectory families

**Date:** 2026-10-04
**Status:** in force
**Cites:** user instruction, 2026-10-04

No v2 trajectory family emits a `Remove`. It remains available in the sampler as an
operation, and `measure_factors` still classifies it correctly if one ever appears, but it
is not part of the benchmark design and adding it requires an explicit design decision
recorded here.

Verified: as of this entry no builder in `generator/trajectories.py` emits `Remove`.

---

## D-007 — Probes must account for every candidate

**Date:** 2026-10-04
**Status:** in force
**Cites:** user instruction, 2026-10-04; SPEC §4

Counterfactual probes were silently lossy: a removal whose replay became invalid was
discarded without being counted, and setup `Put` removals were mixed in with update
removals even though $U$ excludes `Put`. Both defects are fixed.

Every candidate removal now lands in exactly one bucket, and the buckets sum to the
candidate count by construction (asserted, not assumed):

- `valid_answer_changing`
- `valid_answer_preserving`
- `invalid_replay`
- `excluded_setup` — a setup `Put`, skipped because setup intervention is not part of the
  experiment. Opt in with `include_setup=True`.

Selection keeps valid answer-changing and answer-preserving probes in a caller-controlled
balance (`balance=0.5` by default) rather than "prefer sensitive, then insensitive". The
realised balance is reported by `result.realised_balance()`; a degenerate case where one
class is empty is reported, never implied.

Redo-validity probes previously only ever produced the **invalid** class, which made the
condition uninformative. Both classes are now generated deliberately:

- valid redo — `Undo`, then stop; `world.can_redo` is `True`.
- invalid redo — `Undo`, then one further operation that clears the redo stack;
  `world.can_redo` is `False`.

The label is read from `can_redo(history)`, never assumed from the requested class. A
constructed example whose realised label contradicts the requested class raises rather
than being silently relabelled. `build_redo_validity_examples` builds a balanced batch by
construction and enforces the caller's tolerance.

**Distinguishing `None` from invalid:** SPEC §4 gives a removed object the gold answer
`None`. A `None` answer is therefore a *valid* answer and must never be confused with an
invalid replay. The two cases use separate representations.

---

## D-008 — Legacy root modules are removed, not maintained

**Date:** 2026-10-04
**Status:** in force
**Cites:** user decision, 2026-10-04; AGENTS.md §6 rule 10

The repository root carried a second, parallel implementation of sampling and trajectory
building. It is removed.

- `generator.py` and `analysis.py` at the root were already **dead**: the `generator/` and
  `analysis/` packages shadow them on import. Deleted.
- `pipeline.py`, `sampler.py`, `trajectory.py` and `example.py` formed a live legacy chain
  used only by `example.py` and `test/smoke_test.py`. Deleted after `test/smoke_test.py` is
  migrated onto the `generator/` package.

**Rationale:** "boring architecture" (AGENTS.md §6 rule 10) permits one implementation.
Two implementations of trajectory construction is exactly the drift the benchmark is meant
to rule out in its subjects.

---

## D-009 — All five RQs are retained

**Date:** 2026-10-04
**Status:** in force
**Cites:** user decision, 2026-10-04; AGENTS.md §6 rule 7, §7

`rq1_depth`, `rq2_revision`, `rq3_distractor`, `rq4_entity_load` and `rq5_pilot` all stay.
No RQ, factor or validator is deleted to make a task pass. Any future scope cut goes
through `/plan` and a recorded decision in `SPEC.md` §8.

---

## D-010 — `data/` and `results/` are gitignored and regenerated

**Date:** 2026-10-04
**Status:** in force
**Cites:** AGENTS.md §10; user decision 2026-10-04

Datasets and run outputs are reproducible from a recorded seed, so they are not committed.
`data/` and `results/` are gitignored and removed from the index; history retains the old
files, the working tree does not depend on them. `report/` was **restored** from git — the
LaTeX class, `citations.bib` and the 28 figures are source assets, not run outputs.

**Consequence:** `test/test_eval_pipeline.py::test_cli_mock_run_eval` fails until the
datasets are regenerated at the end of the consistency work. This is an expected failure,
not a regression, and must not be "fixed" by weakening the test.

---

## D-011 — Thresholds live in one constants module

**Date:** 2026-10-04
**Status:** in force
**Cites:** AGENTS.md §5

`generator/constants.py` is the single owner of threshold and version literals: $L_{max}$,
$\tau$, curve-model names, prompt versions, container-count bounds, the decoding
configuration, and the `GENERATOR_VERSION` / `RENDERER_VERSION` / `SCORING_VERSION` stamps
that records must carry.

`generator/constants.py` imports nothing from the repository, so `generator/` and `world/`
stay importable without `torch` or `transformers` (AGENTS.md §4).

Entity and container id strings are **not** centralised: they are minted per builder as
`f"o{i}"` / `f"c{i}"` and there is no shared literal to own. Inventing a registry nobody
reads would violate rule 10.

---

## D-012 — The object vocabulary belongs to `render`, and `render` never imports `generator`

**Date:** 2026-10-04
**Status:** in force
**Cites:** AGENTS.md §4, §5

`OBJECT_TYPES` had been moved into `generator/constants.py` and re-exported through
`render/names.py`. That inverted AGENTS.md §5, which names `render/names.py` as the owner of
the object-type vocabulary, and it closed an import cycle:

```
render.names -> generator.constants -> generator/__init__ -> generator.sampler -> render.names
```

so `import render` raised `ImportError: cannot import name 'OBJECT_TYPES' from partially
initialized module 'render.names'`. The failure was invisible because `test/conftest.py`
imports `generator` first, so every test that touches rendering imports the cycle in the one
order that happens to work. A plain `python -c "import render"` still fails.

The vocabulary is restored to `render/names.py`, and `generator/trajectories.py` imports it
from there. `render` has no dependency on `generator` in either direction, which is what
AGENTS.md §5 requires and what makes the layer DAG acyclic. All six top-level packages
(`world`, `render`, `generator`, `analysis`, `eval`, `experiments`) now import standalone.

---

## D-013 — `swap_chain` closes on a `Swap`, so odd $T$ is measurable and `Swap` stays determinative

**Date:** 2026-10-04
**Status:** in force
**Cites:** SPEC §3, AGENTS.md §6 rule 3

`build_swap_chain` emitted target updates as `(Move, Swap)` pairs behind the bound
`updates_done < spec.target_updates - 1`. Each pair contributes two updates, so the loop could
only ever land on even counts: every requested odd $T \ge 3$ measured $T-1$. The RQ5 pilot
sweeps odd depths, so those cells were silently unreachable rather than failing loudly.

Two candidate repairs existed, and the naive one is wrong. Appending a single trailing `Move`
measures the correct $T$ but makes the answer depend on that `Move` instead of on the `Swap`,
which the structural check-5 gate correctly rejects — so the fix would have traded an
unmeasurable factor for a non-determinative required operation.

The adopted construction plans the whole sequence up front and makes the last operation a
`Swap`:

- even $T$: `["pair"] * (T // 2)`
- odd $T$: `["pair"] * ((T - 1) // 2) + ["swap"]`

which yields exactly $T$ target-affecting operations for every valid $T$, keeps the required
`Swap` last, and leaves even-$T$ output byte-identical to the previous implementation.
Verified over 490 instances (6 families x $T \in [1,10]$ x 10 seeds) with zero $T$
mismatches; the only rejections are the families' own documented minimum-$T$ preconditions.

---

## D-014 — Target-effect is decided by differential replay, not by matching operation fields

**Date:** 2026-10-04
**Status:** in force
**Cites:** SPEC §2 rule 2, SPEC §3

The structural target-effect predicate inspected operation dataclass fields and returned `True`
unconditionally for `split`, `merge`, `swap`, `undo` and `redo`. Check 2 (target effect) and
check 5 (no trailing ordinary `Move`) were therefore vacuous: the predicate could not fail.

Field matching is also not decidable here. `Undo` and `Redo` carry no reference to the event
they reverse, so whether they touch the target is only knowable from the surrounding history;
and a `Split` that touches neither the target nor its new child must not count as
target-affecting.

The predicate now reads the target's location before and after the operation from the single
`replay_trace` pass that already exists for gold, narration and probes (SPEC §2 rule 2):

```python
before.location.get(target_obj) != after.location.get(target_obj)
```

Check ordering was also corrected. Because `Move` writes an absolute location, any trailing
target `Move` also satisfies check 3 (necessity), so check 5 was unreachable and every such
instance reported the vaguer "necessity failed". Checks now run presence -> target effect ->
trailing `Move` -> necessity, so the cheapest and most specific diagnosis is reported first.

`split_chain` remains exempt from check 5, and the exemption is now a named constant
(`CHECK5_EXEMPT_FAMILIES`) with its reason recorded rather than an unexplained inline
special case. `Split` creates its child in the source's container, so a later `Move` of that
child is part of the intended chain rather than a competing final determinant.

Three tests in `test/test_structural_causality.py` (`test_target_effect_fires`,
`test_necessity_fires`, `test_no_trailing_move_fires`) had `try/except ValueError: pass`
bodies, so they passed whether or not the check fired; that is why the vacuous predicate
survived a green suite. They are now `pytest.raises(..., match=...)` assertions against
hand-built counterexamples.

---

## D-015 — One implementation per behaviour: shims, re-export copies and stub packages are removed

**Date:** 2026-10-04
**Status:** in force
**Cites:** AGENTS.md §5, §6 rule 10, user instruction 2026-10-04

The user directed that every behaviour have exactly one implementation, and named the
`v2` evaluator as the example. A sweep of the tree found five separate duplications. All are
resolved by deleting the copy and keeping the owner; none was resolved by deleting coverage.

**1. The `v2` evaluator is gone.** `eval/evaluator_v2.py` was a pure re-export shim over
`eval/scoring.py` (it even re-exported the private `_unique_candidate_match`). Nothing
imported it except its own test file. `eval/evaluator_v2.py` and
`test/test_evaluator_v2.py` are deleted; the five instance-extraction tests moved into
`test/test_scoring.py`, where they now also run under `test/run_all.py`. The second
`CANDIDATES` fixture they brought with it is renamed `PROBE_CANDIDATES` so it no longer
shadows the module-level fixture.

**2. `render/templates.py` and `repro/stubs/analysis/` are deleted.** `render/templates.py`
was a byte-for-byte duplicate of `render/narrative.py`: all eight per-operation render
functions (`render_put`, `render_move`, `render_split`, `render_merge`, `render_swap`,
`render_undo`, `render_redo`, `render_remove`) *and* `render_narrative` were defined in both.
`repro/stubs/analysis/` held three-line `raise NotImplementedError("stub")` placeholders
shadowing the real `analysis/first_error.py` (119 lines) and `analysis/failure_onset.py`
(206 lines). Nothing imported the stubs; they existed only to make `sys.path` tricks work and
were a trap for anyone reading `repro/`.

The `F11` regression test imported `render.templates` and `return`ed early on
`ModuleNotFoundError`, so it passed without asserting anything once the module was gone. It
now asserts the duplicate is absent and that `render.narrative.render_narrative` is the real
renderer.

**3. Question templates have one owner.** `question_location`, `question_count` and
`question_counterfactual` were *defined* in `render/names.py` and re-exported by
`render/narrative.py`, inverting AGENTS.md §5; `question_redo_validity` was defined in
**both** files with byte-identical output. All four now live in `render/narrative.py` and
`render/names.py` defines none of them. `generator/instance.py`, which had imported
`question_location` from `render.names`, now imports it from `render.narrative`.

The `F6` regression test asserted `names.question_location is narrative.question_location`,
i.e. it *pinned the wrong direction*. It is rewritten to the stronger property: each helper
exists in `render.narrative`, does not exist at all in `render.names`, and the `render`
package re-export is the identical object.

**4. Extraction has one owner.** `extract_answer` lived in `eval/eval_harness.py` as a
self-described "Deprecated: Thin wrapper" and is now defined in `eval/scoring.py`. It is kept
as a second function rather than folded into `extract_instance_answer` because the contracts
genuinely differ: it accepts a bare `Answer:` prefix and coerces boolean answers to the
candidate set `{"True", "False"}`, whereas `extract_instance_answer` is protocol-strict. Two
documented contracts in one owner module is acceptable; the same function in two modules was
not. `eval_harness.py` re-exports both names.

**5. Stale artifact names are renamed.** `analysis/evaluate_existing_predictions.py` wrote
`evaluator_v2_results/`, `evaluator_v2_predictions.csv`, `evaluator_v2_summary.csv`,
`evaluator_v2_family.csv` and `evaluator_v2_depth.csv`. With the v2 evaluator removed those
names refer to nothing; they are now `evaluation_results/`, `predictions.csv`, `summary.csv`,
`by_family.csv`, `by_depth.csv`.

Deliberately **not** treated as duplication: `eval/engine.py` defines `generate_batch` and
`format_input` three and two times respectively, but on the base, `MockInferenceEngine` and
`HuggingFaceEngine` classes — that is polymorphism, not a second implementation.
`generate.py` (single-instance CLI) and `generate_all.py` (full-grid driver) are distinct
entry points that both now funnel through `generator/instance.py`. `paper/tests/regression/`
tests the LaTeX `acl_natbib.bst` bibliography style and has nothing to do with the Python
suites.

Dead code removed in the same pass: a `while False:` block in `build_swap_chain` and an
`if False:` block in the undo builder (`generator/trajectories.py`), a
`for sop in split_ops: pass` no-op in `generator/trajectory_validation.py`, and a
`try/except Exception: pass` around the frozen-spec `structural_ops` assignment in
`generator/trajectory_specs.py` that could have silently left the invariant unset. The probe
API from D-007 (`CounterfactualProbeSet`, `ProbeAccounting`, `RemovalClassification`,
`RedoValidityExample`, `classify_counterfactual_removals`, `build_redo_validity_examples`) is
now exported from `generator/__init__.py`, so the advertised package surface matches the real
one instead of requiring callers to reach past the package.

---

## D-016 — Three unreviewed changes are adjudicated individually

**Date:** 2026-10-04
**Status:** in force
**Cites:** AGENTS.md §2, §6 rule 11, §9

Three changes reached the working tree without a decision record. Each was tested rather than
accepted or reverted on suspicion.

**1. `experiments/rq2_revision.py` `NUM_CONTAINERS` 3 -> 4: REVERTED to 3.** `SPEC.md` fixes
`num_containers_default: 3`, and no SPEC entry covers RQ2. rq1 uses 3, rq3 uses 4 with a
written reason ("More containers for higher $D$ levels"), rq4 uses 6 for entity load; rq2 had
been silently moved to 4 with no comment. Both values were tested: at
`num_containers` 3 and 4, `generate_instance` over 30 seeds returned 30 valid instances with
zero failures, so 4 bought nothing and the undocumented deviation from the SPEC default was
reverted. Changing a design parameter needs a SPEC decision, not an undocumented edit.

**2. `eval/models.py` OLMo id `allenai/OLMo-2-1B` -> `allenai/OLMo-2-0425-1B`: KEPT.** This
one was a real defect, not drift. Verified against the Hugging Face model card and the
`allenai/OLMo` release configs: OLMo-2 checkpoints are date-stamped
(`OLMo-2-0425-1B`, `OLMo-2-1124-7B`, `OLMo-2-1124-13B`, `OLMo-2-0325-32B`) and no
`allenai/OLMo-2-1B` release exists, so the old id could never have loaded. Still open and
**not** fixed by this decision: AGENTS.md §9 requires every model to be pinned to a commit
hash rather than a floating branch, and `ModelConfig` pins no revision for any model.

**3. `report/main.tex` claim rewrite: REVERTED.** An agent had edited the thesis prose without
being asked to. The file was restored from `HEAD`; it now has no diff.

---

## D-017 — A test that needs a dataset must generate one

**Date:** 2026-10-04
**Status:** in force
**Cites:** D-010, AGENTS.md §6 rule 3

`test/test_eval_pipeline.py::test_cli_mock_run_eval` had been made green by writing twenty
hand-authored JSON records into a temp directory -- a fixture invented to satisfy the CLI
rather than to exercise it. It duplicated a subset of the real record schema and omitted
`trace_hash`, `canonical_trace`, `step_wise_gold`, `sentences` and the three version stamps,
so it would have kept passing even after the schema moved on, and it asserted nothing about
the generator.

The fixture is now twenty real instances from `experiments._common.generate_instance`, at
varying depth ($T \in \{2,3,4,5\}$), written through the same validated path as the sweeps.
The test is strictly stronger: it now fails if the generator, the gate or the record schema
breaks, which is what it was supposed to do. This is the concrete case D-010 anticipated --
absence of `data/` is a reason to generate, never to fabricate.

**6. `L_actual` is removed; `L_word` is the only name.** `MeasuredFactors` stores `L_word`
and additionally exposed an `L_actual` property plus a second `"L_actual"` key in `to_dict()`
carrying the identical value, so every record carried the same number twice under two names.
`L_word` is the SPEC §2 symbol and the quantity the gate reads (`verify_length` compares
`measured.L_word`), so the alias is deleted. Only tests referenced it; the three call sites
were updated. The class docstring's `(E, T, D, U, V, N, L_word)_actual` notation was also
dropped as it mixed the two conventions.

---

## D-018 — Three-RQ structure locked; RQ4/RQ5 merged into RQ1/RQ2

**Date:** 2026-10-04
**Status:** in force
**Cites:** user decision, 2026-10-04; supervisor feedback

The five-RQ structure is collapsed to three RQs with tighter experimental axes:

| Old | New | Disposition |
|---|---|---|
| RQ1: temporal depth | RQ1: Mutation & Depth | Expanded: 7 families × T levels, not just basic_chain |
| RQ2: revision | → RQ1 (revision family) + RQ2 (supersession) | Revision family in RQ1; supersession pairs in RQ2 |
| RQ3: distractor | RQ2: Interference & Supersession | D×N×L crossed with fixed target trajectory |
| RQ4: entity load | → Secondary analysis | No longer an RQ; examined as covariate under RQ1/RQ3 |
| RQ5: structural pilot | → RQ1 (mutation families) | All 5 structural families now part of RQ1 mutation axis |

**Removed files:** `experiments/rq4_entity_load.py`, `experiments/rq5_pilot.py`
**New files:** `experiments/rq1_mutation_depth.py`, `experiments/rq2_interference.py`, `experiments/rq3_scale_reasoning.py`
**Updated:** `generate_all.py` (2 experiments), `SPEC.md` frontmatter `experiments: [RQ1_MUTATION_DEPTH, RQ2_INTERFERENCE, RQ3_SCALE_REASONING]`

Rationale: Each RQ now maps to a distinct experimental axis (mutation structure × depth, interference decomposition, scale × prompting) rather than treating error taxonomy as an RQ. Entity load is a controlled covariate, not a primary question.

---

## Open items carried forward, not yet decided

| Item | Question | Blocks |
|---|---|---|
| `SPEC.md` OPEN-4 / finding F8 | $V$ double-counts: one target-affecting `Undo` contributes to both the revisit count and the history-reversal count. Which term is wrong? | any $V$-derived result; `revision` factor sweep |
| `SPEC.md` OPEN-5 | Two definitions of "target-relevant": state-change (`measure_factors`) vs syntactic (`query_analysis`) | query selection |
| `SPEC.md` OPEN-8 | `eval/models.py::ModelConfig.max_new_tokens = 256` vs the `generate_batch` defaults of 128 in `eval/engine.py`. Which runs? | cross-model comparability |
| `SPEC.md` OPEN-9 | Scoring assumes string candidates; `CountQuery` gold is an `int` | count queries |
| `SPEC.md` OPEN-10 | `TrajectorySpec.revision_count` is never enforced nor checked against measured $V$ | `rq2_revision` |
| `SPEC.md` §2 wording | $T$ is specified as ops that "change the target's **location**"; the code and D-005 say "state **or** location". A `Remove` is $T$ in code but reads as no-change under the literal SPEC wording | SPEC text needs one word changed |
| `SPEC.md` OPEN-3 / F7 | The sampler counts `Put` inside `update_count`; $U$ excludes `Put` | sampler vs factor contract |
| OPEN-7 / F1 | `OBJECT_TYPES` is defined twice, in `generator/trajectories.py` and `render/names.py`, with different contents | name rendering |