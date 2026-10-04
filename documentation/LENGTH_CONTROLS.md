# Length-Matched Controls for RQ1 and RQ3

This specification defines the next generator revision for separating semantic
state updates from rendered narrative length. It does not claim that the
factors are orthogonal: matching `L_word` makes `T` and `L_word` comparable,
but necessarily changes `N` as the filler amount changes.

## Factor conventions

Use the measured fields already written by the generator:

- `T_actual`: post-initialization operations affecting the queried target.
- `D_actual`: post-initialization operations not affecting the queried target.
- `N_actual`: pure narrative sentences with no state transition.
- `L_word`: rendered narrative word count, currently the project token-length
  proxy.

All matching decisions use measured values after rendering, not requested
values. The canonical trace and gold answer remain unchanged by narrative
filler.

## Common matching policy

### Target lengths

For each experiment, first generate a pilot pool using the existing seeds and
rendering path. For each target condition, compute the median `L_word` of the
pool. Set the target length for a matched block to the largest condition
median, rounded to the nearest whole word. This avoids truncating the longest
condition and keeps the target empirically attainable.

Use a separate target-length block for each family, entity count, container
count, and fixed non-length factor. Do not share one global length target
across `basic_chain` and `interleaved_chain`.

### Filler construction

1. Render the canonical operation sentences.
2. Compute `base_words = sum(len(sentence.split()) for sentence in sentences)`.
3. Add pure narrative distractors with `make_distractor_sentences` until the
   result is within the acceptance tolerance of the target.
4. Splice the filler with `splice_distractors`; never alter, remove, or rewrite
   canonical operation sentences.
5. Re-measure `L_word` and `N_actual` from the final sentence list.

The first implementation may use a fixed filler sentence bank and select
sentences by a deterministic seed. Since existing filler sentences have
different lengths, selection should be a bounded search over candidate
subsets, followed by random splice placement. The seed must be recorded.

### Acceptance and failure

Accept a record when:

```text
abs(L_word - target_length) <= 3 words
```

The tolerance is applied to measured `L_word`, not to an estimate. Retry with
another trajectory/render seed when the target cannot be reached within 50
attempts. Preserve failed-attempt counts in the generation summary.

Report the mean, median, standard deviation, minimum, and maximum `L_word` per
condition. Also report the corresponding `N_actual` distribution; length
matching must not hide the induced `T`/`N` trade-off.

## RQ1: temporal depth

### Main sweep

Keep the current design:

```text
family = basic_chain
E = 1
D = 0
T in {2, 4, 6, 8, 12, 16}
```

For each matched block, pad lower-`T` conditions with neutral narrative
filler until they share the block target length. The resulting records should
have `D_actual = 0`, `N_actual >= 0`, and the requested `T` verified by
`T_actual`.

### Length-only control

Add a separate condition for each high-depth target length:

```text
condition = rq1_length_only
family = basic_chain
E = 1
T = 2
D = 0
N = variable filler count chosen to match the T=16 L_word distribution
```

Use the same seed-matching policy where possible: the control and its paired
high-`T` record should share the seed but use independently generated
canonical traces only when sharing a trace would make the conditions
artificially identical. The control tests whether the observed high-depth
decline can be reproduced by length alone. Store `matched_target_length` and
`matched_condition = rq1_T16` in each control record.

The primary comparison is `rq1_T16` versus `rq1_length_only`, both at matched
`L_word`. Secondary comparisons use every `T` level against the same target
length where attainable.

## RQ3: semantic distractors versus narrative noise

Keep the semantic sweep:

```text
family = interleaved_chain
E = 3
T = 8
D in {0, 4, 8, 16}
```

For each `D`, generate a matched narrative-noise condition with the same
`E`, `T`, and target length. The key comparison is:

```text
semantic condition: D = d, N = 0
noise condition:    D = 0, N chosen so L_word matches semantic condition
```

This is the primary matched-length test. It compares state-changing
interference with pure text at approximately equal surface length. Do not use
the current `D=4, N=4` condition as the matched control unless its measured
`L_word` falls within the acceptance tolerance; `N=4` is a starting point,
not a matching guarantee.

For every matched pair, store:

```json
{
  "matched_pair_id": "rq3_d4_seed0001",
  "matched_role": "semantic_distractor|narrative_noise",
  "matched_target_length": 123,
  "length_tolerance": 3
}
```

The semantic condition must retain `N_actual = 0` unless a separate mixed
condition is explicitly being studied. The noise condition must retain
`D_actual = 0`. This keeps the interpretation of the pair clear.

## RQ2 and RQ4 bookkeeping

For RQ2, preserve the existing control/revision pairing and add the same
length matching to each pair. The control and revision records should report
their individual measured `L_word`; do not infer equality from requested
parameters.

For RQ4, retain fixed `T` and `D`, and add `initial_description_words` to
the measured metadata. This is a descriptive covariate for setup length, not
an additional claim of causal isolation.

## Independence decision rule

Before fitting effects, calculate the pairwise Pearson correlation matrix over
the realized dataset for `E_actual`, `T_actual`, `D_actual`, `V_actual`,
`L_word`, and `N_actual`. Flag every pair with `abs(r) > 0.3`.

Any reported factor effect must include:

1. the unadjusted estimate;
2. a logistic model with `L_word` as a covariate, e.g.
   `accuracy ~ factor + L_word + model`;
3. estimates stratified by prespecified `L_word` bins when enough observations
   exist in each bin.

Report both the requested and measured factors. Use language such as
"varies while measuring and reporting coupling" rather than "isolates" or
"orthogonalizes".

## Required generator metadata

Add these optional fields to each matched record without removing existing
fields:

```json
{
  "matched_target_length": 123,
  "length_tolerance": 3,
  "length_match_error": -1,
  "matched_pair_id": "rq3_d4_seed0001",
  "matched_role": "semantic_distractor",
  "filler_seed": 4001
}
```

The existing `measured_factors.L_word` remains the source of truth. A record
must never be labeled matched unless `abs(length_match_error) <=
length_tolerance`.