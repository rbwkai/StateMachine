from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from .errors import InvalidOperation
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
    counterfactual probe). The message names the offending trace index, so a
    rejected ablation is auditable without replaying prefixes by hand.
    """
    state = WorldState(object_type={}, location={}, containers=set(containers), step_index=0)
    hist = history if history is not None else History()
    trace: List[Tuple[Operation, WorldState, WorldState]] = []
    for index, op in enumerate(ops):
        state_before = state
        try:
            state = apply_op(op, state, hist)
        except InvalidOperation as exc:
            raise InvalidOperation(f"op at index {index}: {exc}") from exc
        trace.append((op, state_before, state))
    return trace, state, hist