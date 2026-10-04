
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
                return f"the {idx+1}th {obj_type}"
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
    Generate distractor sentences, EXCLUDING a specific container name
    to prevent answer leakage via elimination.
    """
    sentences = []
    all_containers = list(names.container_names.values())
    if exclude_container and exclude_container in all_containers:
        all_containers.remove(exclude_container)
    unrelated_types = [
        t for t in OBJECT_TYPES if t not in used_object_types
    ] or OBJECT_TYPES

    for _ in range(n):
        if rng.random() < 0.5 and all_containers:
            c = rng.choice(all_containers)
            sentences.append(
                f"{c.capitalize()} {rng.choice(DISTRACTOR_FLAVOR)}."
            )
        else:
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
    
    Uses a more efficient O(n+m) algorithm by generating all insertion positions
    upfront and then building the result in a single pass.
    """
    if not distractor_sentences:
        return list(op_sentences)
    
    n_ops = len(op_sentences)
    n_dist = len(distractor_sentences)
    
    # Generate all insertion positions upfront (between 0 and n_ops + i for i-th distractor)
    # This maintains the same distribution as the original sequential insert
    positions = []
    for i in range(n_dist):
        pos = rng.randint(0, n_ops + i)
        positions.append(pos)
    
    # Sort positions with their distractors to process in order
    indexed_positions = list(zip(positions, distractor_sentences))
    indexed_positions.sort(key=lambda x: x[0])
    
    # Build result in single pass
    result = []
    op_idx = 0
    dist_idx = 0
    
    for pos, dist_sentence in indexed_positions:
        # Add op sentences up to the insertion position
        while op_idx < n_ops and op_idx <= pos - dist_idx:
            result.append(op_sentences[op_idx])
            op_idx += 1
        result.append(dist_sentence)
        dist_idx += 1
    
    # Add remaining op sentences
    while op_idx < n_ops:
        result.append(op_sentences[op_idx])
        op_idx += 1
    
    return result
