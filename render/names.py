from __future__ import annotations

import random
from typing import Dict, Iterable

from world import WorldState

CONTAINER_ADJS = [
    "wooden", "metal", "old", "small", "large", "blue", "red",
    "dusty", "narrow", "tall", "green", "battered", "glass",
]
CONTAINER_NOUNS = [
    "drawer", "cabinet", "box", "shelf", "closet", "chest",
    "bin", "cupboard", "bag", "crate", "trunk", "basket",
]
OBJECT_TYPES = [
    "key", "apple", "book", "coin", "pen", "cup",
    "phone", "ring", "letter", "toy", "candle", "map",
]


class NameRegistry:
    """Assigns stable, human-readable surface names to container ids for
    one generated example, and renders object references (handling the
    'which one do I mean' ambiguity introduced once SPLIT can create two
    same-type objects in the same container).
    """

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
        """Definite-reference phrase for an object, e.g. 'the key' or,
        when a duplicate of the same type currently shares its location
        (post-SPLIT), 'one of the apples'."""
        obj_type = state.object_type[obj_id]
        loc = state.location.get(obj_id)
        if loc is not None:
            siblings = [
                oid for oid, cid in state.location.items()
                if cid == loc and state.object_type[oid] == obj_type
            ]
            if len(siblings) > 1:
                return f"one of the {obj_type}s"
        return f"the {obj_type}"