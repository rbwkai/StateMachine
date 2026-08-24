"""
generator/metadata.py
=====================
Measured factor computation for DWS-Bench trajectories.

Every generated trajectory reports **requested** parameters (via TrajectorySpec)
and **measured** parameters (via this module).

Measured factor tuple:
----------------------
    (E, T, D, V, L_word, N)_actual = f(canonical trace, rendered narrative)

Where:
    (E, T, D, V) <- derived from canonical symbolic trace replay:
        E_actual : Total unique entity IDs instantiated across trajectory lifetime.
        T_actual : Post-init operations that change the queried target's state.
        D_actual : Post-init operations NOT changing the queried target's state.
        V_actual : Genuine location revisits by the target after intervening moves.

    (L_word, N) <- derived from rendered narrative text:
        L_word   : Rendered narrative word count (proxy for tokenizer token count).
        N_actual : Count of pure textual distractor sentences (0 state transitions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

from world import (
    History,
    Merge,
    Move,
    Operation,
    Put,
    Redo,
    Split,
    Swap,
    Undo,
    WorldState,
    apply_op,
)
from world.errors import InvalidOperation


# ============================================================
# MeasuredFactors
# ============================================================

@dataclass
class MeasuredFactors:
    """
    Independently measured experimental factors for one trajectory instance:
        (E, T, D, V, L_word, N)_actual
    """

    # Number of unique entities ever existing in the trajectory.
    E_actual: int

    # Post-init ops changing the target entity's state/location.
    T_actual: int

    # Post-init ops NOT changing the target entity's state/location.
    D_actual: int

    # Number of genuine target-location revisits.
    V_actual: int

    # Rendered narrative word count proxy (-1 if not rendered yet).
    L_word: int = -1

    # Count of pure textual distractor sentences.
    N_actual: int = 0

    @property
    def L_actual(self) -> int:
        """Alias for L_word for backwards compatibility."""
        return self.L_word

    def to_dict(self) -> dict:
        return {
            "E_actual": self.E_actual,
            "T_actual": self.T_actual,
            "D_actual": self.D_actual,
            "V_actual": self.V_actual,
            "L_word": self.L_word,
            "L_actual": self.L_word,
            "N_actual": self.N_actual,
        }


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
    and analyzing the rendered narrative.

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

    # --------------------------------------------------------
    # 1. Unique entity count (E_actual)
    # --------------------------------------------------------
    entity_ids: Set[str] = set()

    for op in ops:
        if isinstance(op, Put):
            entity_ids.add(op.obj_id)
        elif isinstance(op, Split):
            entity_ids.add(op.source_obj_id)
            entity_ids.add(op.new_obj_id)

    E_actual = len(entity_ids)

    # --------------------------------------------------------
    # 2. T_actual, D_actual via canonical replay
    # --------------------------------------------------------
    state = WorldState(
        object_type={},
        location={},
        containers=containers,
        step_index=0,
    )
    history = History()

    T_actual = 0
    D_actual = 0
    target_locations_sequence: List[str] = []

    for op in ops:
        # Put ops are setup initialization (not counted in T or D).
        if isinstance(op, Put):
            state = apply_op(op, state, history)
            if op.obj_id == target_obj:
                target_locations_sequence.append(op.container)
            continue

        target_loc_before = state.location.get(target_obj)
        state = apply_op(op, state, history)
        target_loc_after = state.location.get(target_obj)

        target_affected = (target_loc_before != target_loc_after)

        if target_affected:
            T_actual += 1
            if target_loc_after is not None:
                target_locations_sequence.append(target_loc_after)
        else:
            D_actual += 1

    # --------------------------------------------------------
    # 3. V_actual — genuine revisit count
    # --------------------------------------------------------
    V_actual = _count_revisits(target_locations_sequence)

    # --------------------------------------------------------
    # 4. L_word & N_actual — narrative properties
    # --------------------------------------------------------
    if sentences is not None:
        L_word = sum(len(s.split()) for s in sentences)
        # If textual_distractor_count was not explicitly passed, infer from sentence vs op count
        if textual_distractor_count > 0:
            N_actual = textual_distractor_count
        else:
            N_actual = max(0, len(sentences) - len(ops))
    else:
        L_word = -1
        N_actual = textual_distractor_count

    return MeasuredFactors(
        E_actual=E_actual,
        T_actual=T_actual,
        D_actual=D_actual,
        V_actual=V_actual,
        L_word=L_word,
        N_actual=N_actual,
    )


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
# Verification
# ============================================================

def verify_factors(
    requested_E: int,
    requested_T: int,
    requested_D: int,
    measured: MeasuredFactors,
    family: str,
    instance_id: str = "",
    tolerance_V: bool = True,
) -> None:
    """
    Assert that measured factors match requested specification.
    """
    prefix = f"[{instance_id}] " if instance_id else ""

    if measured.E_actual != requested_E:
        raise AssertionError(
            f"{prefix}Measured E_actual={measured.E_actual} != requested E={requested_E} (family={family!r})"
        )

    if measured.T_actual != requested_T:
        raise AssertionError(
            f"{prefix}Measured T_actual={measured.T_actual} != requested T={requested_T} (family={family!r})"
        )

    if measured.D_actual != requested_D:
        raise AssertionError(
            f"{prefix}Measured D_actual={measured.D_actual} != requested D={requested_D} (family={family!r})"
        )

    if not tolerance_V and family == "revision":
        if measured.V_actual < 1:
            raise AssertionError(
                f"{prefix}Measured V_actual={measured.V_actual} < 1 for revision family — no revisit detected"
            )
