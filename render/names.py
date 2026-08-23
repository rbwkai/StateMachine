
from __future__ import annotations

import random
from typing import Dict, Iterable, Sequence, List

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
        loc = state.location.get(obj_id)
        if loc is not None:
            siblings = [
                oid for oid, cid in state.location.items()
                if cid == loc and state.object_type[oid] == obj_type
            ]
            if len(siblings) > 1:
                return f"one of the {obj_type}s"
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
) -> List[str]:
    sentences = []
    all_containers = list(names.container_names.values())
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
                f"Someone mentioned that {t}s have become harder to find lately."
            )
    return sentences


def splice_distractors(
    rng: random.Random,
    op_sentences: Sequence[str],
    distractor_sentences: Sequence[str],
) -> List[str]:
    combined = list(op_sentences)
    for sentence in distractor_sentences:
        combined.insert(rng.randint(0, len(combined)), sentence)
    return combined


def question_location(obj_id: str, state: WorldState, names: NameRegistry) -> str:
    # obj_id is intentionally not exposed in the surface form.
    return f"Where is the {state.object_type[obj_id]} now?"


def question_count(container_id: str, obj_type: str, names: NameRegistry) -> str:
    return (
        f"How many {obj_type}s are in "
        f"{names.container(container_id)} now?"
    )


def question_redo_validity() -> str:
    return (
        "If someone tried to redo the last undone action right now, "
        "would that succeed?"
    )


def question_counterfactual(
    removed_sentence: str,
    obj_id: str,
    state: WorldState,
) -> str:
    obj_type = state.object_type[obj_id]
    return (
        f'Suppose this had not happened: "{removed_sentence}" '
        f"Where would the {obj_type} be now?"
    )
