from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from .operations import Operation, apply_op
from .state import History, WorldState


def replay_trace(
    ops: Sequence[Operation],
    containers: Set[str],
    history: Optional[History] = None,
) -> Tuple[List[Tuple[Operation, WorldState, WorldState]], WorldState, History]:
    """Re-applies ops from an empty world, one at a time.

    Returns (trace, final_state, final_history) where trace is a list of
    (op, state_before, state_after) triples -- one shared replay pass that
    the renderer, step-wise scorer, and counterfactual probes all reuse,
    so narration and gold answers can never drift apart.

    Raises InvalidOperation if any op in the sequence is not valid given
    the state that precedes it (e.g. after removing an earlier step for a
    counterfactual probe).
    """
    state = WorldState(object_type={}, location={}, containers=set(containers), step_index=0)
    hist = history if history is not None else History()
    trace: List[Tuple[Operation, WorldState, WorldState]] = []
    for op in ops:
        state_before = state
        state = apply_op(op, state, hist)
        trace.append((op, state_before, state))
    return trace, state, hist