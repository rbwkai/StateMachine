
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

CONTAINER_ADJS = [
    "wooden", "metal", "old", "small", "large", "blue", "red",
    "dusty", "narrow", "tall", "green", "battered", "glass",
]
CONTAINER_NOUNS = [
    "drawer", "cabinet", "box", "shelf", "closet", "chest",
    "bin", "cupboard", "bag", "crate", "trunk", "basket",
]
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
        same_type_objs = sorted(
            [oid for oid, t in state.object_type.items() if t == obj_type]
        )
        if len(same_type_objs) > 1:
            idx = same_type_objs.index(obj_id)
            if idx == 0:
                return f"the original {obj_type}"
            elif idx == 1:
                return f"the duplicate {obj_type}"
            else:
                # Proper ordinal suffix
                n = idx + 1
                if 10 <= n % 100 <= 20:
                    suffix = "th"
                else:
                    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                return f"the {n}{suffix} {obj_type}"
        return f"the {obj_type}"


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
    """
    if not distractor_sentences:
        return list(op_sentences)
    
    result = list(op_sentences)
    for dist_sentence in distractor_sentences:
        pos = rng.randint(0, len(result))
        result.insert(pos, dist_sentence)
    
    return result
    
    return result
