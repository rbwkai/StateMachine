# Length-Matched Controls

This document specifies length matching as a *control mechanism* for RQ1 and RQ2.
It does not claim that any factor is orthogonal: matching `L_word` makes `T` and
`L_word` comparable, but necessarily changes `N` as the filler amount changes.

**Status (2026-10-05): specified, not implemented.** The design below is
consistent with the three-RQ structure locked in D-018. The generator currently has
no length-matching machinery. `experiments/rq2_interference.py` declares
`LENGTH_MATCHED = True`, but no filler search, no `matched_*` metadata, and no
per-condition length target exist. The corresponding test,
`test_experiment_scripts.py::test_failsnow_rq2_cells_are_length_matched_within_three_words`,
is red for exactly this reason.

---

## Factor conventions

Use the measured fields the generator already writes:

- `T_actual`: post-initialization operations affecting the queried target.
- `D_actual`: post-initialization operations not affecting the queried target.
- `N_actual`: pure narrative sentences with no state transition.
- `L_word`: rendered narrative word count. This is the project length proxy; the
  ceiling is `L_MAX_WORDS = 600` from `generator/constants.py`.
- `L_tok`: tokenizer diagnostic, recorded by `eval/engine.py`, never a generation
  gate.

All matching decisions use measured values after rendering, never requested ones.
The canonical trace and gold answer are unchanged by narrative filler.

---

## Common matching policy

### Target lengths

For each experiment, first generate a pilot pool using existing seeds and the
rendering path. For each target condition compute the median `L_word` of the pool.
Set a matched block's target length to the largest condition median, rounded to a
whole word. This avoids truncating the longest condition and keeps the target
empirically attainable.

Use a separate target-length block per family, entity count, container count, and
fixed non-length factor. Do not share one global length target across
`basic_chain` and `interleaved_chain`.

### Filler construction

1. Render the canonical operation sentences.
2. Compute `base_words = sum(len(sentence.split()) for sentence in sentences)`.
3. Add pure narrative distractors with `make_distractor_sentences` until the result
   is within the acceptance tolerance of the target.
4. Splice the filler with `splice_distractors`. Never alter, remove, or rewrite a
   canonical operation sentence.
5. Re-measure `L_word` and `N_actual` from the final sentence list.

Because existing filler sentences differ in length, selection must be a bounded
search over candidate subsets followed by seeded splice placement. The filler seed
is recorded.

Note the interaction with the answer-leakage gate: filler sentences currently live
in the only region that gate inspects, so longer `N` increases the chance of a
leak rejection. Length matching and `SPEC.md` PARTIAL-10 must be resolved together.

### Acceptance and failure

```text
abs(L_word - target_length) <= 3 words
```

The tolerance applies to measured `L_word`, not an estimate. Retry with another
trajectory/render seed when the target cannot be reached within the attempt
budget; `generate_instance_with_retry` defaults to 50 attempts. Preserve
failed-attempt counts in the generation summary.

Report mean, median, standard deviation, minimum, and maximum `L_word` per
condition, plus the corresponding `N_actual` distribution. Length matching must not
hide the induced `T`/`N` trade-off.

---

## RQ1: mutation and depth

RQ1 varies `T` across seven families with `D = 0` and `N = 0`:

```text
family ∈ {basic_chain, revision, split_chain, merge_chain, swap_chain,
          undo_chain, undo_redo_chain}
E per family as in experiments/rq1_mutation_depth.py
D = 0
T per family as in RQ1_FAMILIES
```

Length matching is **not** applied inside RQ1. RQ1 is the sweep that establishes
depth difficulty, and padding its low-`T` cells with filler would confound the very
factor being measured. If RQ1 needs a length control, it belongs as a separate
condition set, not as a modification of the depth cells.

### Length-only control (optional extension)

A separate condition per high-depth target length:

```text
condition = rq1_length_only
family = basic_chain
E = 1
T = 2
D = 0
N = filler count chosen to match the T=16 L_word distribution
```

Share the seed with the paired high-`T` record where that does not make the
conditions identical. The control asks whether a high-depth decline is reproducible
by length alone. Store `matched_target_length` and `matched_condition = rq1_T16`.

Primary comparison: `rq1_T16` vs `rq1_length_only` at matched `L_word`. Secondary:
every `T` level against the same target where attainable.

---

## RQ2: interference and supersession

RQ2 holds the target trajectory fixed at `interleaved_chain`, E=3, T=8, 4
containers, and crosses `D` and `N`.

### Current grid

```text
D ∈ {4, 8, 16} at N = 0
N ∈ {4, 8, 16} at D = 4
revision T = 8 (supersession)
```

`D = 0` is not in the current grid; the `basic_chain T=8` cell is treated as the
`D=0` baseline. Seven conditions at 50 instances is 350 records.

### Required change: length-matched noise conditions

The current `D` sweep at `N=0` and the `N` sweep at `D=4` are *not* length matched,
so any difference between them confounds interference with surface length. Replace
the raw `N` cells with paired cells:

```text
semantic condition: D = d, N = 0
noise condition:    D = 0, N chosen so L_word matches the semantic condition
```

This is the primary matched-length test: state-changing interference against pure
text at approximately equal surface length. Do not treat `N = 4` as a control
unless its measured `L_word` lands inside the tolerance.

The superseded-information arm (revision T=8) pairs with the D=4, N=0 cell and
should also be length matched.

For every matched pair, store:

```json
{
  "matched_pair_id": "rq2_d4_seed0001",
  "matched_role": "semantic_distractor|narrative_noise",
  "matched_target_length": 123,
  "length_tolerance": 3,
  "length_match_error": -1,
  "filler_seed": 4001
}
```

The semantic condition must retain `N_actual = 0` unless a mixed condition is being
studied explicitly. The noise condition must retain `D_actual = 0`.

### Entity load

`E` is a covariate, not an RQ (D-018). If `E` is examined, hold `T` and `D` fixed
and report `initial_description_words` as a descriptive covariate for setup length,
not as an additional causal isolation.

---

## Independence decision rule

Before fitting any factor effect, compute the pairwise Pearson correlation matrix
over the realized dataset for `E_actual`, `T_actual`, `D_actual`, `V_actual`,
`L_word`, and `N_actual`. Flag every pair with $|r| > 0.3$.

Every reported factor effect must include:

1. the unadjusted estimate;
2. a logistic model with `L_word` as a covariate, e.g.
   `accuracy ~ factor + L_word + model`;
3. estimates stratified by prespecified `L_word` bins, where each bin has enough
   observations.

Report requested and measured factors side by side. Use "varies while measuring and
reporting coupling" rather than "isolates" or "orthogonalizes".

`V` is not independent of `T` by construction (RESOLVED-3): a revisit is a subtype
of target-relevant history. Any `V` analysis must be conditional on `T` bins or use
matched pairs with identical $(T, D, U)$.

---

## Required generator metadata

Add these optional fields to each matched record without removing existing fields:

```json
{
  "matched_target_length": 123,
  "length_tolerance": 3,
  "length_match_error": -1,
  "matched_pair_id": "rq2_d4_seed0001",
  "matched_role": "semantic_distractor",
  "filler_seed": 4001
}
```

`measured_factors.L_word` remains the source of truth. A record must never be
labelled matched unless `abs(length_match_error) <= length_tolerance`.