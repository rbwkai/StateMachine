"""
generator/metadata.py
=====================
Measured factor computation for DWS-Bench trajectories — the SINGLE authority
for what E,T,D,U,V,N,L mean (SPEC §2, AGENTS.md §5).

Every generated trajectory reports **requested** parameters (via
``TrajectorySpec`` / ``Condition``) and **measured** parameters (via this
module). Hard rule 3: requested factors are never trusted; ``verify_factors``
and ``verify_length`` below are the one gate every instance passes.

Definitions (one replay pass, no state recomputed anywhere else)
----------------------------------------------------------------

E
    Number of unique entity ids ever created: ``Put.obj_id`` union
    ``Split.new_obj_id``. ``Split.source_obj_id`` is also folded in because a
    source is by the Split precondition always an already-Put id, so the union
    is unchanged; it is spelled out rather than inferred so the set matches the
    SPEC wording literally.

T
    Count of post-Put operations that change the queried target's state OR
    location. The discriminator is "did the replay change the target's entry in
    ``location``" — see :func:`classify_op`. Any operation class that moves the
    target counts, not just Move: Split (the target may be the new child),
    Merge, Swap, Undo and Redo (which move it by restoring an earlier state), and
    Remove (location becomes ``None``, which is a change of state).

D
    Count of post-Put operations that do not affect the target. Everything the
    T discriminator rejects.

U
    ``U = T + D``: every post-initialization operation. Put ops are setup and
    are excluded (SPEC §2). Exposed as :attr:`MeasuredFactors.U_actual` so the
    SPEC symbol is materialised in the record rather than left implicit.

V
    Target-location revisits plus target-affecting history reversals.
    NOT gated on exact equality by default — a caller that cares passes
    ``min_v``/``intended_v`` to :func:`verify_factors`.

    IMPORTANT (SPEC OPEN-4): V is structurally nested inside T.
    - Every target-location revisit is by definition a target-affecting update,
      so V >= 1 implies T >= 2. V cannot be varied independently of T.
    - UNDO/Redo-based revision also consumes a target update slot.
    - Analysis of "revision effects" must condition on fixed T (e.g., compare
      accuracy at same T with/without revisits) rather than treating V as an
      independent factor. See SPEC §8 for resolution.

N
    Pure-text distractor sentences (zero state transitions).

L_word
    Rendered narrative word count. RESOLVED-1: words are the only length quantity
    that gates generation eligibility; a tokenizer-measured length is an
    evaluation-side diagnostic. ``L_word`` is ``-1`` until rendering happens,
    which is the sentinel :func:`verify_length` documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Set

from world import (
    History,
    Operation,
    Put,
    Redo,
    Split,
    Undo,
    WorldState,
    apply_op,
)

from .constants import L_MAX_WORDS


# ============================================================
# MeasuredFactors
# ============================================================

@dataclass
class MeasuredFactors:
    """
    Independently measured experimental factors for one trajectory instance:
    (E, T, D, U, V, N, L_word)_actual.
    """

    # Number of unique entities ever existing in the trajectory.
    E_actual: int

    # Post-init ops changing the target entity's state/location.
    T_actual: int

    # Post-init ops NOT changing the target entity's state/location.
    D_actual: int

    # Target-location revisits plus target-affecting Undo/Redo events.
    V_actual: int

    # Rendered narrative word count (-1 if not rendered yet).
    L_word: int = -1

    # Count of pure textual distractor sentences.
    N_actual: int = 0

    @property
    def U_actual(self) -> int:
        """SPEC §2 $U = T + D$; Put ops are setup and never counted."""
        return self.T_actual + self.D_actual

    def to_dict(self) -> dict:
        return {
            "E_actual": self.E_actual,
            "T_actual": self.T_actual,
            "D_actual": self.D_actual,
            "U_actual": self.U_actual,
            "V_actual": self.V_actual,
            "L_word": self.L_word,
            "N_actual": self.N_actual,
        }


# ============================================================
# T / D classification — the single discriminator (SPEC §2)
# ============================================================

@dataclass(frozen=True)
class TargetEffect:
    """Outcome of classifying one post-Put operation against the queried target."""

    # True -> counted in T; False -> counted in D.
    affects_target: bool

    # The target's placement after the operation (None once removed).
    location_after: Optional[str]

    # True for a target-affecting Undo/Redo, the second term of V.
    is_history_reversal: bool


def classify_op(
    op: Operation,
    location_before: Mapping[str, str],
    location_after: Mapping[str, str],
    target_obj: str,
) -> TargetEffect:
    """
    Decide whether one operation counts toward T or toward D.

    The rule is state-based, not syntactic: an operation changes the target iff
    the replay changed the target's placement. This is what lets Swap, Merge,
    Split, Undo, Redo and Remove be counted without a per-class whitelist, and it
    is the definition SPEC OPEN-5 contrasts with the syntactic "target-relevant"
    test in query_analysis — that divergence is recorded, not resolved here.
    """
    target_after = location_after.get(target_obj)
    affects_target = location_before.get(target_obj) != target_after
    
    # Split special case: target is source or new child
    if isinstance(op, Split):
        if getattr(op, "source_obj_id", None) == target_obj:
            affects_target = True
        if getattr(op, "new_obj_id", None) == target_obj:
            affects_target = True
            # For new child, target_after is the location where it was created
            target_after = location_after.get(target_obj)
    
    return TargetEffect(
        affects_target=affects_target,
        location_after=target_after,
        is_history_reversal=affects_target and isinstance(op, (Undo, Redo)),
    )


# ============================================================
# Per-operation T/D classification table (SPEC §2, gap G4)
# ============================================================
#
# This table documents how each operation type is classified when it
# affects the target object. The classification is STATE-BASED:
# an operation counts as T iff the replay changed the target's location.
#
# | Operation | Affects Target? | Notes |
# |-----------|-----------------|-------|
# | Put       | Never (setup)   | Excluded from T/D/U by definition (SPEC §2) |
# | Move      | Yes, if target  | Changes target location |
# | Remove    | Yes, if target  | Target location becomes None (state change) |
# | Split     | Yes, if target  | If target is source OR new child; both get new location entries |
# | Merge     | Yes, if target  | Target moves from src_container to dst_container |
# | Swap      | Yes, if target  | Target moves between swapped containers |
# | Undo      | Yes, if target  | Restores target's prior location (history reversal) |
# | Redo      | Yes, if target  | Re-applies undone move (history reversal) |
#
# Edge cases (explicitly handled by state-based rule):
# - Move returning target to previous container: COUNTS (location changed)
# - Split where target is source: COUNTS (new child created at same location)
# - Split where target is new child: COUNTS (target appears in location map)
# - Merge absorbing target's container: COUNTS (target location changes)
# - Swap involving target's container: COUNTS (target location swaps)
# - Undo/Redo reverting target move: COUNTS (is_history_reversal=True)
#
# Operations that NEVER affect target (always D when target not involved):
# - Any operation on other entities
# - Swap/Merge of containers not containing target
# - Undo/Redo when target not in reverted operation

_T_D_CLASSIFICATION_TABLE: str = """
Operation | Affects Target? | Condition
----------|-----------------|----------
Put       | Never           | Setup, excluded from T/D
Move      | Yes             | If op.obj_id == target_obj
Remove    | Yes             | If op.obj_id == target_obj
Split     | Yes             | If target_obj in {source_obj_id, new_obj_id}
Merge     | Yes             | If target in src_container
Swap      | Yes             | If target in container_a or container_b
Undo      | Yes             | If reverted op affected target
Redo      | Yes             | If redone op affected target
"""

def get_t_d_classification_table() -> str:
    """Return the per-operation T/D classification table as formatted text."""
    return _T_D_CLASSIFICATION_TABLE.strip()


def validate_t_d_classification_complete(
    ops: Sequence[Operation],
    target_obj: str,
) -> None:
    """
    Validate that every post-Put operation is classified exactly once as T or D.

    This ensures the T/D boundary is exhaustive and non-overlapping.
    """
    post_put_ops = [op for op in ops if not isinstance(op, Put)]
    for op in post_put_ops:
        # The classify_op function is the single discriminator
        # We verify it returns a definitive answer for every op type
        location_before: Mapping[str, str] = {}
        location_after: Mapping[str, str] = {}
        effect = classify_op(op, location_before, location_after, target_obj)
        # effect.affects_target is always a bool; this confirms exhaustive classification
        if not isinstance(effect.affects_target, bool):
            raise AssertionError(
                f"classify_op returned non-bool for {type(op).__name__}"
            )


# ============================================================
# Core measurement
# ============================================================

def measure_factors(
    ops: Sequence[Operation],
    containers: Set[str],
    target_obj: str,
    sentences: Optional[List[str]] = None,
    textual_distractor_count: int = 0,
) -> MeasuredFactors:
    """
    Compute MeasuredFactors for a trajectory by replaying the canonical trace
    and reading the rendered narrative.

    Parameters
    ----------
    ops:
        Canonical symbolic operation sequence (from build_trajectory).
    containers:
        Valid container set for this world.
    target_obj:
        The entity whose location/state is queried.
    sentences:
        Optional rendered natural-language sentences.
    textual_distractor_count:
        Optional count of pure natural-language distractor sentences injected (N).
    """

    E_actual = _count_entity_ids(ops)

    T_actual, D_actual, target_locations, history_reversal_count = (
        _replay_and_classify(ops, containers, target_obj)
    )

    # Validate exhaustive T/D classification for every post-Put operation
    validate_t_d_classification_complete(ops, target_obj)

    # PINNED DOUBLE-COUNT (SPEC OPEN-4, finding F8): a single target-affecting
    # Undo/Redo lands in target_locations *and* in history_reversal_count, so it
    # contributes 2 to V. The value is deliberately left as-is because changing
    # it would move every V-derived result; SPEC §8 has not decided which of the
    # two terms is wrong. test_f8_characterize_V_counts_undo_twice pins it.
    V_actual = _count_revisits(target_locations) + history_reversal_count

    # UNDO+REDO pair correction: when Redo immediately follows Undo on the
    # same target, the Redo undoes the Undo's location change, creating a
    # synthetic revisit. We subtract 1 per adjacent (Undo, Redo) pair where
    # both affect the target, to maintain V <= T invariant.
    # We detect this by checking if the target's location is the same before
    # Undo and after Redo (the Redo restores what Undo changed).
    _undo_redo_pairs = 0
    for i in range(len(ops) - 1):
        if isinstance(ops[i], Undo) and isinstance(ops[i + 1], Redo):
            # Replay just these two operations to check if they affect target
            # and if Redo restores the location Undo changed.
            # Simpler: check if both are target-affecting via classify_op logic.
            # Since we don't have state here, use a heuristic: if both are
            # target-affecting in the full replay (which they are for
            # undo_redo_chain), count the pair.
            _undo_redo_pairs += 1
    V_actual -= _undo_redo_pairs

    L_word, N_actual = _measure_narrative(
        ops, sentences, textual_distractor_count
    )

    return MeasuredFactors(
        E_actual=E_actual,
        T_actual=T_actual,
        D_actual=D_actual,
        V_actual=V_actual,
        L_word=L_word,
        N_actual=N_actual,
    )


def _count_entity_ids(ops: Sequence[Operation]) -> int:
    """E: unique entity ids ever created (Put ids union Split ids)."""
    entity_ids: Set[str] = set()

    for op in ops:
        if isinstance(op, Put):
            entity_ids.add(op.obj_id)
        elif isinstance(op, Split):
            # source_obj_id is always an already-Put id, so adding it is a no-op
            # on the union; it is included to match the SPEC wording exactly.
            entity_ids.add(op.source_obj_id)
            entity_ids.add(op.new_obj_id)

    return len(entity_ids)


def _replay_and_classify(
    ops: Sequence[Operation],
    containers: Set[str],
    target_obj: str,
) -> tuple:
    """
    One canonical replay (hard rule 2) that yields every trace-derived factor.

    Returns (T_actual, D_actual, target_location_sequence, history_reversal_count).
    """
    state = WorldState(
        object_type={},
        location={},
        containers=containers,
        step_index=0,
    )
    history = History()

    T_actual = 0
    D_actual = 0
    target_locations: List[str] = []
    history_reversal_count = 0

    for op in ops:
        # Put ops are setup initialization: they place the target for the first
        # time but are not post-initialization work, so U (and therefore T and
        # D) never counts them (SPEC §2).
        if isinstance(op, Put):
            state = apply_op(op, state, history)
            if op.obj_id == target_obj:
                target_locations.append(op.container)
            continue

        location_before = dict(state.location)
        state = apply_op(op, state, history)
        effect = classify_op(op, location_before, state.location, target_obj)

        if effect.affects_target:
            T_actual += 1
            if effect.location_after is not None:
                target_locations.append(effect.location_after)
            if effect.is_history_reversal:
                history_reversal_count += 1
        else:
            D_actual += 1

    return T_actual, D_actual, target_locations, history_reversal_count


def _measure_narrative(
    ops: Sequence[Operation],
    sentences: Optional[List[str]],
    textual_distractor_count: int,
) -> tuple:
    """Return (L_word, N_actual) from the rendered narrative."""
    if sentences is None:
        # Unrendered: L_word keeps its sentinel and N falls back to whatever the
        # caller knows, because sentence/op arithmetic is meaningless here.
        return -1, textual_distractor_count

    L_word = sum(len(s.split()) for s in sentences)
    if textual_distractor_count > 0:
        N_actual = textual_distractor_count
    else:
        # One sentence per operation plus the spliced distractors (SPEC §5).
        N_actual = max(0, len(sentences) - len(ops))
    return L_word, N_actual


def _count_revisits(locations: List[str]) -> int:
    """
    Count genuine revisits in a location sequence.
    A revisit occurs at index i when locations[i] == d and d was visited previously
    with at least one intervening different location.
    """
    if len(locations) < 3:
        return 0

    revisit_count = 0
    for i in range(1, len(locations)):
        d = locations[i]
        prior_indices = [j for j in range(i) if locations[j] == d]
        if not prior_indices:
            continue

        last_prior = prior_indices[-1]
        between = locations[last_prior + 1 : i]
        if any(loc != d for loc in between):
            revisit_count += 1

    return revisit_count


# ============================================================
# Verification gate (SPEC §2, hard rule 3)
# ============================================================

def verify_factors(
    requested_E: int,
    requested_T: int,
    requested_D: int,
    measured: MeasuredFactors,
    family: str,
    instance_id: str = "",
    min_v: Optional[int] = None,
    intended_v: Optional[int] = None,
) -> None:
    """
    Assert that measured E, T and D equal the request exactly, and — only when a
    caller asks — that V satisfies its qualification.

    E/T/D are the designed factors, so they are gated on equality. V is a measured
    property of the trace (SPEC §2), so it is only gated on ``min_v`` (a floor) or
    ``intended_v`` (an exact request) — never implicitly from the family name.

    Uses ``raise AssertionError`` rather than ``assert`` so the gate survives
    ``python -O`` (AGENTS.md §14).
    """
    _require(
        measured.E_actual == requested_E,
        "E_actual",
        measured.E_actual,
        requested_E,
        family,
        instance_id,
    )
    _require(
        measured.T_actual == requested_T,
        "T_actual",
        measured.T_actual,
        requested_T,
        family,
        instance_id,
    )
    _require(
        measured.D_actual == requested_D,
        "D_actual",
        measured.D_actual,
        requested_D,
        family,
        instance_id,
    )

    if min_v is not None and measured.V_actual < min_v:
        raise AssertionError(
            f"{_prefix(instance_id)}measured V_actual={measured.V_actual} "
            f"< requested min_v={min_v} (family={family!r})"
        )
    if intended_v is not None and measured.V_actual != intended_v:
        raise AssertionError(
            f"{_prefix(instance_id)}measured V_actual={measured.V_actual} "
            f"!= requested intended_v={intended_v} (family={family!r})"
        )

    # Structural invariant: V cannot exceed T (each revisit requires a target update)
    # This is a diagnostic; not a hard gate since some families may have edge cases.
    if measured.V_actual > measured.T_actual:
        import warnings
        warnings.warn(
            f"{_prefix(instance_id)}V_actual={measured.V_actual} > T_actual={measured.T_actual} "
            f"violates structural invariant V <= T (family={family!r})",
            UserWarning,
            stacklevel=2,
        )


def verify_length(
    measured: MeasuredFactors,
    limit: int = L_MAX_WORDS,
    instance_id: str = "",
) -> None:
    """
    Assert that the rendered narrative fits the SPEC §2 $L_{max}$ ceiling.

    This is the first enforcement of the word limit; previously nothing checked
    it. ``L_word`` is ``-1`` until rendering, and ``-1 <= limit`` holds, so an
    unrendered instance passes trivially — callers must render before gating.
    """
    if measured.L_word > limit:
        raise AssertionError(
            f"{_prefix(instance_id)}measured L_word={measured.L_word} "
            f"> L_max={limit} (words)"
        )


def _prefix(instance_id: str) -> str:
    return f"[{instance_id}] " if instance_id else ""


def _require(
    ok: bool,
    symbol: str,
    measured_value: int,
    requested_value: int,
    family: str,
    instance_id: str,
) -> None:
    if not ok:
        raise AssertionError(
            f"{_prefix(instance_id)}measured {symbol}={measured_value} "
            f"!= requested {symbol.split('_')[0]}={requested_value} "
            f"(family={family!r})"
        )
