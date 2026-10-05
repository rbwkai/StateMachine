
from __future__ import annotations

import random
from typing import Dict, Iterable, Sequence, List, Tuple

from world import WorldState

# Object-type vocabulary is owned here per AGENTS.md §5. `render` must never
# import from `generator`: `generator/__init__` pulls in the sampler, which
# imports this module, so a reverse import closes a cycle on `import render`.
OBJECT_TYPES: Tuple[str, ...] = (
    "key", "apple", "phone", "map", "coin", "book", "pen",
    "card", "ring", "token", "gem", "watch", "letter",
)

# Canonical Undo sentence. Owned here (the lower render layer) because both the
# renderer and splice_distractors() need it and `render.narrative` imports this
# module, not the other way round. render_undo() renders this exact string, so
# there is one definition of the sentence template (AGENTS.md §5).
UNDO_SENTENCE = "That last action was undone."

CONTAINER_ADJS = [
    "wooden", "metal", "old", "small", "large", "blue", "red",
    "dusty", "narrow", "tall", "green", "battered", "glass",
]
CONTAINER_NOUNS = [
    "drawer", "cabinet", "box", "shelf", "closet", "chest",
    "bin", "cupboard", "bag", "crate", "trunk", "basket",
]


def ordinal(n: int) -> str:
    """English ordinal for a positive integer: 1st, 2nd, 3rd, 11th, 21st."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def pluralize_object(obj_type: str) -> str:
    if obj_type.endswith(("s", "x", "z", "ch", "sh")):
        return f"{obj_type}es"
    if obj_type.endswith("y") and len(obj_type) > 1 and obj_type[-2] not in "aeiou":
        return f"{obj_type[:-1]}ies"
    return f"{obj_type}s"


class NameRegistry:
    def __init__(self, rng: random.Random, containers: Iterable[str]):
        self.rng = rng
        self.container_names: Dict[str, str] = {}
        used = set()
        for cid in sorted(containers):
            while True:
                name = f"the {rng.choice(CONTAINER_ADJS)} {rng.choice(CONTAINER_NOUNS)}"
                if name not in used:
                    used.add(name)
                    break
            self.container_names[cid] = name

    def container(self, container_id: str) -> str:
        return self.container_names[container_id]

    def obj(self, obj_id: str, state: WorldState) -> str:
        obj_type = state.object_type[obj_id]
        # Creation order, not lexicographic id order: Put/Split insert into
        # state.object_type in trace order, so dict order is the order the
        # reader meets the objects. Sorting ids would name "o10" the 3rd key.
        same_type_objs = [
            oid for oid, t in state.object_type.items() if t == obj_type
        ]
        if len(same_type_objs) > 1:
            return self.obj_at(obj_type, same_type_objs.index(obj_id) + 1)
        return f"the {obj_type}"

    def obj_at(self, obj_type: str, rank: int) -> str:
        """Name for the ``rank``-th (1-based) object of a type.

        Same ladder as obj() -- original, duplicate, then ordinals -- but
        addressable without a WorldState, so a Put can name the object it
        introduces before the state contains it (render_narrative passes the
        creation rank it derived from the trace).
        """
        if rank == 1:
            return f"the original {obj_type}"
        if rank == 2:
            return f"the duplicate {obj_type}"
        return f"the {ordinal(rank)} {obj_type}"


DISTRACTOR_FLAVOR = [
    "has a faint smell of cedar",
    "creaks when opened",
    "was recently dusted",
    "sits near a window",
    "was a gift from a relative",
    "has a small scratch on one side",
    "is slightly heavier than it looks",
    "was bought many years ago",
]


def make_distractor_sentences(
    rng: random.Random,
    n: int,
    names: NameRegistry,
    used_object_types: Sequence[str],
    exclude_container: str = None,
) -> List[str]:
    """
    Generate distractor sentences that do not name any containers,
    to prevent answer leakage via elimination.
    """
    sentences = []
    unrelated_types = [
        t for t in OBJECT_TYPES if t not in used_object_types
    ] or OBJECT_TYPES

    for _ in range(n):
        t = rng.choice(unrelated_types)
        sentences.append(
            f"Someone mentioned that {pluralize_object(t)} have become harder to find lately."
        )
    return sentences


def splice_distractors(
    rng: random.Random,
    op_sentences: Sequence[str],
    distractor_sentences: Sequence[str],
) -> List[str]:
    """
    Splice distractor sentences into op sentences at random positions.

    Uses sequential insertion to maintain the correct distribution:
    each distractor is inserted at a uniformly random position in the
    current sequence (which grows by one each time).

    A distractor is never placed directly in front of the Undo sentence: the
    reader would have to decide whether the inserted sentence was part of the
    story, and the sentence the Undo inverts must stay adjacent to it. Appending
    at the end stays available, so the candidate set is never empty and the
    position distribution is unchanged when no Undo sentence is present.
    """
    if not distractor_sentences:
        return list(op_sentences)
    
    result = list(op_sentences)
    for dist_sentence in distractor_sentences:
        positions = [p for p in range(len(result) + 1) if p == len(result)
                     or result[p] != UNDO_SENTENCE]
        result.insert(positions[rng.randrange(len(positions))], dist_sentence)
    
    return result
    
    return result
