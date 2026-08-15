from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrajectorySpec:
    """
    Explicit structural specification for one trajectory family.

    Setup operations:
        Create the requested number of entities.

    Updates:
        target_updates + distractor_updates

    Therefore:

        total_updates =
            target_updates + distractor_updates

    Put operations are setup and are NOT included in
    total_updates.

    General numerical consistency is checked here.

    Family-specific structural constraints are checked by
    trajectory_validation.validate_trajectory().
    """

    family: str

    # --------------------------------------------------------
    # World complexity
    # --------------------------------------------------------

    entity_count: int

    num_containers: int = 3

    # --------------------------------------------------------
    # Temporal complexity
    # --------------------------------------------------------

    total_updates: int = 1

    target_updates: int = 1

    distractor_updates: int = 0

    # --------------------------------------------------------
    # Interleaving
    # --------------------------------------------------------

    min_interleaving: float = 0.0

    # --------------------------------------------------------
    # Revision
    # --------------------------------------------------------

    revision_count: int = 0

    min_unique_target_locations: int = 1

    # --------------------------------------------------------
    # Optional deterministic target
    # --------------------------------------------------------

    target_obj: Optional[str] = None

    def __post_init__(self):
        # ====================================================
        # General validation
        # ====================================================

        if self.family not in {
            "basic_chain",
            "interleaved_chain",
            "revision",
        }:
            raise ValueError(
                f"unknown trajectory family: "
                f"{self.family!r}"
            )

        if self.entity_count < 1:
            raise ValueError(
                "entity_count must be >= 1"
            )

        if self.num_containers < 2:
            raise ValueError(
                "num_containers must be >= 2"
            )

        if self.total_updates < 1:
            raise ValueError(
                "total_updates must be >= 1"
            )

        if self.target_updates < 1:
            raise ValueError(
                "target_updates must be >= 1"
            )

        if self.distractor_updates < 0:
            raise ValueError(
                "distractor_updates must be >= 0"
            )

        if not 0.0 <= self.min_interleaving <= 1.0:
            raise ValueError(
                "min_interleaving must be in [0, 1]"
            )

        if self.revision_count < 0:
            raise ValueError(
                "revision_count must be >= 0"
            )

        if self.min_unique_target_locations < 1:
            raise ValueError(
                "min_unique_target_locations "
                "must be >= 1"
            )

        # ====================================================
        # Update-count consistency
        # ====================================================

        expected_updates = (
            self.target_updates
            + self.distractor_updates
        )

        if expected_updates != self.total_updates:
            raise ValueError(
                "total_updates must equal "
                "target_updates + distractor_updates; "
                f"got total_updates={self.total_updates}, "
                f"target_updates={self.target_updates}, "
                f"distractor_updates={self.distractor_updates}"
            )

        # ====================================================
        # IMPORTANT:
        #
        # Do NOT put family-specific constraints here.
        #
        # For example, do not reject:
        #
        #   basic_chain + distractor_updates=1
        #
        # here.
        #
        # Such constraints belong to
        # validate_trajectory().
        #
        # This keeps specification from being confused with
        # trajectory validation and allows the validation gate
        # to actually be tested.
        # ====================================================