"""
Checklist 3 — Shortcut and solvability audits.

The property under test is *anti-shortcut*: a solver that ignores order,
distractors or causality must score at or below chance. Solvers here read the
symbolic trace (or the rendered sentences) exactly once per instance; none of
them is allowed to peek at the gold beyond the final comparison.

Chance is derived from the real answer space of the cell (number of containers,
or the count range), never hard-coded to 1/3.

Tests named ``test_failsnow_*`` encode the CORRECT property for defects recorded
in the pre-freeze audit, so they fail today by design.
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest  # noqa: E402

from test.conftest import build_record, containers_of, ops_from_record, quiet  # noqa: E402

SEEDS = range(30)

# (family, T, E, D, N) cells the released grid actually uses (SPEC §7).
LOCATION_CELLS = [
    ("basic_chain", 4, 1, 0, 0),
    ("basic_chain", 16, 1, 0, 0),
    ("interleaved_chain", 4, 2, 2, 0),
    ("interleaved_chain", 8, 2, 2, 0),
    ("revision", 4, 1, 0, 0),
    ("revision", 8, 1, 0, 0),
    ("revision", 16, 1, 0, 0),
    ("undo_chain", 4, 1, 0, 0),
    ("undo_chain", 8, 1, 0, 0),
    ("undo_redo_chain", 8, 1, 0, 0),
    ("undo_redo_chain", 16, 1, 0, 0),
]
COUNT_CELLS = [
    ("split_chain", 4, 2, 0, 0),
    ("split_chain", 8, 2, 0, 0),
]

# 30 seeds per cell: the standard error at chance (p=1/3) is about 0.09, so a
# 0.20 margin is ~2 SE. Anything above that is a shortcut, not sampling noise.
CHANCE_MARGIN = 0.20


def records_for(family, entity_count, target_updates, distractor_updates=0, n_textual=0, seeds=SEEDS):
    with quiet():
        out = [
            build_record(
                family, seed,
                entity_count=entity_count,
                target_updates=target_updates,
                distractor_updates=distractor_updates,
                textual_distractors=n_textual,
                max_attempts=20,
            )
            for seed in seeds
        ]
    return [r for r in out if r is not None]


def name_to_id(record):
    return {v: k for k, v in record["final_state"]["container_display_names"].items()}


# ---------------------------------------------------------------------------
# solvers (each returns a container id, or None if it cannot answer)
# ---------------------------------------------------------------------------

def solve_last_move(record):
    target = record["query_entity"]
    answer = None
    for step in record["canonical_trace"]:
        if step["op_type"] == "MOVE" and step.get("obj_id") == target:
            answer = step["dst"]
    return answer


def solve_penultimate_move(record):
    target = record["query_entity"]
    seen = [
        s["dst"] for s in record["canonical_trace"]
        if s["op_type"] == "MOVE" and s.get("obj_id") == target
    ]
    return seen[-2] if len(seen) >= 2 else None


def solve_stateless(record):
    """The container the target was first put in; no later sentence is read."""
    target = record["query_entity"]
    for step in record["canonical_trace"]:
        if step["op_type"] == "PUT" and step.get("obj_id") == target:
            return step["container"]
    return None


def solve_parity(record):
    """Alternating-structure guess: the location visited at parity T % k."""
    target = record["query_entity"]
    visits = []
    for step in record["canonical_trace"]:
        if step["op_type"] == "PUT" and step.get("obj_id") == target:
            visits.append(step["container"])
        elif step["op_type"] == "MOVE" and step.get("obj_id") == target:
            visits.append(step["dst"])
    if not visits:
        return None
    return visits[record["measured_factors"]["T_actual"] % len(visits)]


_MOVED_TO = re.compile(r"was moved to (.+?)\.?$")


def solve_regex_last_moved_to(record):
    """Last sentence matching 'was moved to <container>' -- text only."""
    id_of = name_to_id(record)
    answer = None
    for sentence in record["sentences"]:
        match = _MOVED_TO.search(sentence)
        if match:
            found = id_of.get(match.group(1).rstrip("."))
            if found:
                answer = found
    return answer


def solve_most_mentioned(record):
    id_of = name_to_id(record)
    counts = collections.Counter()
    for sentence in record["sentences"]:
        for display, cid in id_of.items():
            if display in sentence:
                counts[cid] += 1
    return counts.most_common(1)[0][0] if counts else None


def solve_eliminate_distractors(record, n_textual):
    """Drop containers that only distractor sentences mention, then guess."""
    id_of = name_to_id(record)
    traced = set()
    target = record["query_entity"]
    for step in record["canonical_trace"]:
        for field in ("container", "dst", "src_container", "dst_container",
                      "container_a", "container_b"):
            if field in step and (field == "container" or step.get("obj_id", target) == target):
                traced.add(step[field])
    only_distractor = {cid for cid, display in id_of.items()
                       if cid not in traced and any(display in s for s in record["sentences"])}
    remaining = [cid for cid in id_of if cid not in only_distractor]
    if len(remaining) != 1:
        return None
    return remaining[0]


def accuracy(records, solver):
    if not records:
        return 0.0, 0
    correct = 0
    for record in records:
        prediction = solver(record)
        if prediction is not None and prediction == record["gold_container"]:
            correct += 1
    return correct / len(records), len(records)


def chance_for(records):
    """1 / number of containers actually present in the cell."""
    if not records:
        return 1.0
    containers = len(records[0]["final_state"]["containers"])
    return 1.0 / containers


# ---------------------------------------------------------------------------
# shortcut solvers must not beat chance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "family,t,e,d,n", LOCATION_CELLS,
    ids=[f"{f}-T{t}" for f, t, _e, _d, _n in LOCATION_CELLS],
)
def test_failsnow_no_solver_beats_chance(family, t, e, d, n):
    """[checklist 3] 'Run each solver per family, T and N cell, and assert
    accuracy <= chance + margin.'

    [fails now] expected: every order-blind solver stays at chance.
    currently: the last-Move destination is exactly right for basic_chain,
    interleaved_chain and revision (1.00 vs chance 0.33), the penultimate
    destination is exactly right for undo_chain and undo_redo_chain, and the
    stateless first-Put guess is right for revision at T=4 and T=16 -- so those
    cells are solvable without tracking state at all.
    """
    records = records_for(family, e, t, d, n)
    assert records, f"no instances for {family} T={t}"
    chance = chance_for(records)
    offenders = {}
    for name, solver in (
        ("last_move", solve_last_move),
        ("penultimate_move", solve_penultimate_move),
        ("stateless", solve_stateless),
        ("parity", solve_parity),
        ("regex_last_moved_to", solve_regex_last_moved_to),
    ):
        acc, n = accuracy(records, solver)
        if acc > chance + CHANCE_MARGIN:
            offenders[name] = acc
    assert not offenders, (
        f"{family} T={t}: shortcuts above chance {chance:.2f} (n={len(records)}): "
        + ", ".join(f"{k}={v:.2f}" for k, v in offenders.items())
    )


@pytest.mark.parametrize(
    "family,t,e,d,n",
    [("interleaved_chain", 4, 2, 2, 4), ("basic_chain", 4, 1, 0, 8)],
    ids=["interleaved-N4", "basic-N8"],
)
def test_distractor_elimination_does_not_solve_the_cell(family, t, e, d, n):
    """[checklist 3] 'Elimination by distractors' is one of the listed solvers.

    Expected: dropping the containers that only distractor sentences mention
    leaves more than one candidate, so the solver stays at chance.
    """
    records = records_for(family, e, t, d, n)
    assert records, f"no instances for {family} T={t} N={n}"
    chance = chance_for(records)
    acc, count = accuracy(
        records, lambda r: solve_eliminate_distractors(r, n)
    )
    assert acc <= chance + CHANCE_MARGIN, (
        f"{family} T={t} N={n}: distractor elimination scored {acc:.2f} "
        f"against chance {chance:.2f} (n={count})"
    )


@pytest.mark.parametrize(
    "family,t,e,d,n", [("revision", 4, 1, 0, 0), ("revision", 16, 1, 0, 0),
                       ("undo_chain", 4, 1, 0, 0)],
    ids=["revision-T4", "revision-T16", "undo-T4"],
)
def test_failsnow_most_mentioned_container_is_not_the_answer(family, t, e, d, n):
    """[checklist 3] 'Most-mentioned container, and elimination by distractors.'

    [fails now] expected: the container named most often in the narrative is not
    the gold container more often than chance. currently: in revision at T=4 and
    T=16 the most-mentioned container is the gold container on every instance.
    """
    records = records_for(family, e, t, d, n)
    assert records, f"no instances for {family} T={t}"
    chance = chance_for(records)
    acc, count = accuracy(records, solve_most_mentioned)
    assert acc <= chance + CHANCE_MARGIN, (
        f"{family} T={t}: most-mentioned container scored {acc:.2f} against "
        f"chance {chance:.2f} (n={count})"
    )


@pytest.mark.parametrize(
    "family,t,e,d,n", COUNT_CELLS, ids=[f"{f}-T{t}" for f, t, _e, _d, _n in COUNT_CELLS]
)
def test_failsnow_majority_class_does_not_solve_count_cells(family, t, e, d, n):
    """[checklist 3] 'Majority class per cell.'

    [fails now] expected: the most frequent gold answer is a minority guess, so
    majority-class accuracy stays near chance over the count answer space.
    currently: at T=4 the gold count is '1' for every seed, so majority-class
    accuracy is 1.00 while the count answer space is {1, 2}.
    """
    records = records_for(family, e, t, d, n)
    assert records
    counts = collections.Counter(r["gold_answer"] for r in records)
    # Count answers range over 0..(E + splits) by construction (SPEC §2), so the
    # answer space is a property of the cell, not of the sample that survived the
    # gate -- otherwise a degenerate cell would define its own chance level.
    space = e + 2
    chance = 1.0 / space
    majority = counts.most_common(1)[0][1] / len(records)
    assert majority <= chance + CHANCE_MARGIN, (
        f"{family} T={t}: majority class {counts.most_common(1)[0][0]!r} covers "
        f"{majority:.2f} of {len(records)} instances (answer space {sorted(counts)}, "
        f"chance {chance:.2f})"
    )


# ---------------------------------------------------------------------------
# distribution of gold answers
# ---------------------------------------------------------------------------

def chi_square_gold_balance(records):
    """Pearson chi-square of the gold container counts against uniform."""
    ids = sorted(records[0]["final_state"]["containers"])
    counts = collections.Counter(r["gold_container"] for r in records)
    k = len(ids)
    if k < 2 or len(records) < 10:
        return 0.0, counts
    expected = len(records) / k
    return sum((counts.get(cid, 0) - expected) ** 2 for cid in ids) / expected, counts


@pytest.mark.parametrize(
    "family,t,e,d,n", LOCATION_CELLS,
    ids=[f"{f}-T{t}" for f, t, _e, _d, _n in LOCATION_CELLS],
)
def test_failsnow_gold_container_is_balanced_across_containers(family, t, e, d, n):
    """[checklist 3] 'length-only and position-only bias, with a chi-square test
    of gold balance across containers or counts.'

    [fails now] expected: the gold container is distributed uniformly over the
    cell's containers (chi-square < 9.21, df=2, p=0.01). currently:
    interleaved_chain pins the target in one container for the whole cell (40/40
    in the same container at T=4 and again at T=8), so container identity alone
    identifies the answer.
    """
    records = records_for(family, e, t, d, n)
    assert records, f"no instances for {family} T={t}"
    stat, counts = chi_square_gold_balance(records)
    assert stat < 9.21, (
        f"{family} T={t}: gold container is unbalanced (chi2={stat:.1f}, df=2) "
        f"counts={dict(counts)} n={len(records)}"
    )


@pytest.mark.parametrize(
    "family,t,e,d,n", LOCATION_CELLS,
    ids=[f"{f}-T{t}" for f, t, _e, _d, _n in LOCATION_CELLS],
)
def test_gold_container_is_one_the_target_visited(family, t, e, d, n):
    """A sanity invariant that must hold even for a degenerate cell."""
    records = records_for(family, e, t, d, n)
    assert records
    for record in records:
        visited = set()
        target = record["query_entity"]
        for step in record["canonical_trace"]:
            if step["op_type"] == "PUT" and step.get("obj_id") == target:
                visited.add(step["container"])
            elif step["op_type"] == "MOVE" and step.get("obj_id") == target:
                visited.add(step["dst"])
        assert record["gold_container"] in visited


# ---------------------------------------------------------------------------
# ablations: structural necessity and playability
# ---------------------------------------------------------------------------

def query_for(record):
    from generator.probes import CountQuery, LocationQuery

    target = record["query_entity"]
    if record["gold_answer"].isdigit():
        obj_type = next(
            s["obj_type"] for s in record["canonical_trace"]
            if s["op_type"] == "PUT" and s.get("obj_id") == target
        )
        return CountQuery(record["gold_container"], obj_type)
    return LocationQuery(target)


def classify(record):
    from generator import classify_counterfactual_removals

    return classify_counterfactual_removals(
        ops_from_record(record), containers_of(record), query_for(record)
    )


@pytest.mark.parametrize(
    "family,e,t", [("split_chain", 2, 4), ("swap_chain", 2, 4), ("undo_chain", 1, 4),
                   ("undo_redo_chain", 1, 8)],
    ids=["split", "swap", "undo", "undo_redo"],
)
def test_failsnow_every_required_op_is_necessary_on_the_asked_query(family, e, t):
    """[checklist 3] 'For every required structural op, removing each occurrence
    (not just the last) must change the answer.'

    [fails now] expected: dropping any occurrence of the family's required op
    changes the answer to the question that is actually asked. currently:
    split_chain's Split is answer-preserving for the rendered question (the
    asked container still holds one gem), because the validator compares counts
    in the Merge destination instead of the queried container.
    """
    from generator.dataset_spec import REQUIRED_STRUCTURAL_OPS

    required = REQUIRED_STRUCTURAL_OPS[family]
    records = records_for(family, e, t, seeds=range(10))
    assert records, f"no instances for {family}"
    offenders = []
    for record in records:
        cls = classify(record)
        ops = ops_from_record(record)
        answer_changing = {p["remove_step"] for p in cls.answer_changing}
        for index, op in enumerate(ops):
            if type(op).__name__.lower() in required and index not in answer_changing:
                offenders.append((record["instance_id"], index, type(op).__name__))
    assert not offenders, (
        f"{family}: required ops that do not change the asked answer: {offenders[:6]}"
    )


@pytest.mark.parametrize(
    "family,e,t", [("split_chain", 2, 4), ("swap_chain", 2, 4), ("undo_chain", 1, 4),
                   ("undo_redo_chain", 1, 8)],
    ids=["split", "swap", "undo", "undo_redo"],
)
def test_no_unplayable_counterfactual_is_accepted_as_causal(family, e, t):
    """[checklist 3] The validator must not accept 'removing this op breaks the
    replay' as evidence of causality: an unplayable ablation is untested, not
    proven necessary. Families whose dependencies force this are named in
    UNPLAYABLE_COUNTERFACTUAL_ALLOWED and are excluded by design."""
    from generator.dataset_spec import REQUIRED_STRUCTURAL_OPS
    from generator.structural import UNPLAYABLE_COUNTERFACTUAL_ALLOWED

    if family in UNPLAYABLE_COUNTERFACTUAL_ALLOWED:
        pytest.skip(f"{family} is a declared unplayable-counterfactual family")
    required = REQUIRED_STRUCTURAL_OPS[family]
    records = records_for(family, e, t, seeds=range(10))
    assert records, f"no instances for {family}"
    offenders = []
    for record in records:
        cls = classify(record)
        ops = ops_from_record(record)
        for index in cls.invalid_indices:
            if type(ops[index]).__name__.lower() in required:
                offenders.append((record["instance_id"], index))
    assert not offenders, (
        f"{family}: required ops whose ablation is unplayable but counted as "
        f"causal: {offenders[:6]}"
    )


@pytest.mark.parametrize(
    "family,e,t", [("swap_chain", 2, 4), ("undo_chain", 1, 4), ("undo_redo_chain", 1, 8)],
    ids=["swap", "undo", "undo_redo"],
)
def test_probe_accounting_covers_every_removal(family, e, t):
    """Hard rule 8: invalid probe replays are retained in accounting."""
    records = records_for(family, e, t, seeds=range(5))
    assert records, f"no instances for {family}"
    for record in records:
        cls = classify(record)
        classified = (
            len(cls.answer_changing) + len(cls.answer_preserving) + len(cls.invalid_indices)
        )
        assert classified + len(cls.excluded_setup_indices) == cls.candidates, (
            f"{record['instance_id']}: {cls.candidates} removals, {classified} "
            f"classified, {len(cls.excluded_setup_indices)} setup"
        )


# ---------------------------------------------------------------------------
# crafted bad traces must be rejected
# ---------------------------------------------------------------------------

def _split_spec(**kwargs):
    from dataclasses import replace

    from test.conftest import make_spec

    return replace(make_spec("split_chain", 2, 4), **kwargs)


def test_failsnow_validator_rejects_a_split_that_does_not_change_the_asked_answer():
    """A Split whose child never reaches the queried container is not causal.

    Trace: the target is split, merged away, then moved out of the queried
    container, so the asked count is 1 with the Split and 1 without it -- yet the
    validator compares counts in the Merge destination (2 vs 0) and accepts the
    trace. The Split is then reported as structurally necessary while the
    rendered question is answerable without it.
    """
    from world import Merge, Move, Put, Split
    from generator import validate_structural_causality

    containers = {"c0", "c1", "c2"}
    ops = [
        Put("o0", "gem", "c0"),
        Put("o1", "key", "c1"),
        Split("o0", "o0b"),
        Merge("c0", "c1"),
        Move("o0", "c2"),          # asked container is c2: one gem with or without the Split
    ]
    _, state, _ = __import__("world").replay_trace(ops, containers)
    from generator.probes import CountQuery

    without_split = [ops[0], ops[1], ops[3], ops[4]]
    _, state_without, _ = __import__("world").replay_trace(without_split, containers)
    assert state.location["o0"] == "c2"
    assert state_without.location["o0"] == "c2"
    # the asked question (count of gems in c2) is unchanged -> Split is not causal
    assert CountQuery("c2", "gem").read(state) == CountQuery("c2", "gem").read(state_without)
    with pytest.raises(ValueError):
        validate_structural_causality(ops, containers, "o0", _split_spec())


def test_validator_rejects_an_irrelevant_split():
    """Split of two distractor objects leaves the target untouched."""
    from world import Merge, Move, Put, Split
    from generator import validate_structural_causality

    containers = {"c0", "c1", "c2"}
    ops = [
        Put("o0", "gem", "c0"),
        Put("o1", "key", "c1"),
        Put("o2", "pen", "c1"),
        Split("o1", "o1b"),
        Move("o0", "c2"),
        Merge("c1", "c0"),
    ]
    with pytest.raises(ValueError):
        validate_structural_causality(ops, containers, "o0", _split_spec())


def test_validator_rejects_a_trailing_plain_move_as_the_final_determinant():
    """check 5: the last determinant may not be an ordinary Move."""
    from world import Merge, Move, Put
    from generator import validate_structural_causality
    from test.conftest import make_spec
    from dataclasses import replace

    spec = replace(make_spec("merge_chain", 2, 4), query_type="count")
    containers = {"c0", "c1", "c2"}
    ops = [
        Put("o0", "gem", "c1"),
        Put("o1", "gem", "c1"),
        Merge("c1", "c0"),
        Move("o0", "c2"),          # trailing ordinary move decides the answer
    ]
    with pytest.raises(ValueError) as excinfo:
        validate_structural_causality(ops, containers, "o0", spec)
    assert "trailing" in str(excinfo.value), str(excinfo.value)


def test_validator_rejects_an_irrelevant_undo():
    """An Undo of a distractor's move does not change the target's answer."""
    from world import Move, Put, Split, Undo
    from generator import validate_structural_causality

    containers = {"c0", "c1", "c2"}
    ops = [
        Put("o0", "gem", "c0"),
        Put("o1", "key", "c1"),
        Move("o1", "c2"),          # distractor move, then undone
        Undo(),                    # irrelevant undo
        Split("o0", "o0b"),
    ]
    with pytest.raises(ValueError):
        validate_structural_causality(ops, containers, "o0", _split_spec())
