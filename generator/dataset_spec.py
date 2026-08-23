from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict


# ============================================================
# Capability Groupings (5-way Taxonomy)
# ============================================================

class CapabilityGroup(Enum):
    """
    Theoretical taxonomy of benchmark capabilities:

    A. SEQUENTIAL_STATE_TRACKING (RQ1, RQ4)
        Basic linear chain progression and state revision.
        Families: basic_chain, revision.
        Targeted by Naturalistic Transfer (ProPara-CRTS).

    B. MULTI_ENTITY_INTERFERENCE (RQ3)
        Selective tracking of target entity under distractor moves.
        Families: interleaved_chain.
        Targeted by Naturalistic Transfer (ProPara-CRTS).

    C. IDENTITY_TRANSFORMATION (RQ5)
        Dynamic modification of entity set cardinality (branching & fusion).
        Families: split_chain, merge_chain.
        Synthetic extension benchmark.

    D. GLOBAL_STATE_OPERATIONS (RQ5)
        Simultaneous multi-entity bilateral updates.
        Families: swap_chain.
        Synthetic extension benchmark.

    E. TEMPORAL_EDIT_HISTORY (RQ5)
        Mutable history rollback, re-application, and contradiction resolution.
        Families: undo_chain, undo_redo_chain.
        Synthetic procedural execution / state-machine extension.
    """

    SEQUENTIAL_STATE_TRACKING = auto()
    MULTI_ENTITY_INTERFERENCE = auto()
    IDENTITY_TRANSFORMATION = auto()
    GLOBAL_STATE_OPERATIONS = auto()
    TEMPORAL_EDIT_HISTORY = auto()


FAMILY_TO_CAPABILITY_GROUP: Dict[str, CapabilityGroup] = {
    "basic_chain": CapabilityGroup.SEQUENTIAL_STATE_TRACKING,
    "revision": CapabilityGroup.SEQUENTIAL_STATE_TRACKING,
    "interleaved_chain": CapabilityGroup.MULTI_ENTITY_INTERFERENCE,
    "split_chain": CapabilityGroup.IDENTITY_TRANSFORMATION,
    "merge_chain": CapabilityGroup.IDENTITY_TRANSFORMATION,
    "swap_chain": CapabilityGroup.GLOBAL_STATE_OPERATIONS,
    "undo_chain": CapabilityGroup.TEMPORAL_EDIT_HISTORY,
    "undo_redo_chain": CapabilityGroup.TEMPORAL_EDIT_HISTORY,
}


def family_capability_group(family: str) -> CapabilityGroup:
    """Return the theoretical capability group for a given trajectory family."""
    try:
        return FAMILY_TO_CAPABILITY_GROUP[family]
    except KeyError as exc:
        raise ValueError(f"Unknown family: {family!r}") from exc


# ============================================================
# Experiment registry
# ============================================================

class Experiment(Enum):
    """
    Top-level experiment grouping.

    CORE
        The primary RQ1–4 benchmark grid: basic_chain,
        interleaved_chain, revision.

    DEPTH
        Depth calibration sweep: same families as CORE across a
        wider T range to identify the operational T-window before
        generating the final CORE grid.

    STRUCTURAL
        RQ5 — How does model performance change when dynamic world-state
        reasoning is extended beyond ordinary sequential location updates to
        identity transformations, global state operations, and temporal edit history?
        Sub-themes:
            - Identity Transformation (split_chain, merge_chain)
            - Global State Operations (swap_chain)
            - Temporal Edit History (undo_chain, undo_redo_chain)
        Scope: single-T pilot (one T level × 5 families × 100
        instances), run alongside Depth calibration.

    NATURALISTIC_TRANSFER
        Evaluation of state tracking on natural process narratives
        derived from ProPara-CRTS.
        Supported canonical operations:
            PUT / CREATE, MOVE, REMOVE / DESTROY, UNCHANGED
        Primary families evaluated:
            basic_chain, interleaved_chain, revision
        Purpose:
            Test whether controlled synthetic state-tracking performance
            transfers to naturally occurring process narratives without
            forcing synthetic extensions into the naturalistic corpus.
    """

    CORE = auto()
    DEPTH = auto()
    STRUCTURAL = auto()
    NATURALISTIC_TRANSFER = auto()


# ============================================================
# Generation status
# ============================================================

class GenerationStatus(Enum):
    """
    Lifecycle status of a benchmark condition.

    GENERATED
        Instances have been generated and are on-disk.

    PENDING_CALIBRATION
        Waiting for a calibration pass (e.g. Depth calibration)
        to determine the concrete parameter values (e.g. which
        T level to use for the single-T pilot).

    PENDING_GENERATION
        Parameters are frozen; generation has not yet run.
    """

    GENERATED = auto()
    PENDING_CALIBRATION = auto()
    PENDING_GENERATION = auto()


# ============================================================
# Condition
# ============================================================

@dataclass(frozen=True)
class Condition:
    """
    A single fully-specified benchmark condition.

    Parameter Formalism
    -------------------
    E : Lifetime Entity Count
        Total number of unique entities instantiated across the trajectory
        lifetime (via initial Put or subsequent Split).
        Initial entity placements = E (or 1 for split_chain where child spawns later).

    U : Post-Initialization Operation Count (U = T + D)
        Count of post-initialization state-changing events in the trajectory
        (Move, Merge, Swap, Undo, Redo). Initial Put operations are
        strictly initialization and are excluded from U.
        Note: While U measures temporal step depth in Groups A & B,
        in Groups C–E the qualitative reasoning complexity is governed by
        operation semantics (branching, bilateral exchange, rollback)
        rather than raw step count alone.

    T : Target-Relevant Operations
        Count of post-init operations that causally alter the target entity's
        state or location.

    D : Distractor Operations (D = U - T)
        Count of post-init operations on non-target entities.

    S : Total Symbolic Operations (S = Initial Placements + U)
        The total count of discrete operations applied to the symbolic simulator.

    L : Rendered Token Length
        Token count of the rendered linguistic narrative (constrained to L < 600
        in the benchmark design). Distinct from sentence count and symbolic step count S.
    """

    family: str
    T: int
    E: int
    D: int
    experiment: Experiment
    generation_status: GenerationStatus = (
        GenerationStatus.PENDING_CALIBRATION
    )

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError("T must be >= 1")
        if self.E < 1:
            raise ValueError("E must be >= 1")
        if self.D < 0:
            raise ValueError("D must be >= 0")
        if self.D >= self.T:
            raise ValueError(
                "D must be < T "
                "(at least one target update required)"
            )

    @property
    def capability_group(self) -> CapabilityGroup:
        return family_capability_group(self.family)

    @property
    def initial_placements(self) -> int:
        """Count of initial Put setup operations."""
        return 1 if self.family == "split_chain" else self.E

    @property
    def total_updates(self) -> int:
        """Post-initialization updates U = T + D."""
        return self.T + self.D

    @property
    def total_transitions(self) -> int:
        """Total state transitions S = Initial Placements + U."""
        return self.initial_placements + self.total_updates


# ============================================================
# Convenience: structural family names
# ============================================================

STRUCTURAL_FAMILIES: frozenset[str] = frozenset({
    "split_chain",
    "merge_chain",
    "swap_chain",
    "undo_chain",
    "undo_redo_chain",
})
