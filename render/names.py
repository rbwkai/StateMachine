
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


_ORDINAL_WORDS: Tuple[str, ...] = (
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth",
)
_TENS_CARDINAL = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
                  6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
_TENS_ORDINAL = {2: "twentieth", 3: "thirtieth", 4: "fortieth", 5: "fiftieth",
                 6: "sixtieth", 7: "seventieth", 8: "eightieth", 9: "ninetieth"}


def ordinal_word(n: int) -> str:
    """Spelled English ordinal: first, second, ..., twenty-first, ...

    Object names carry no digits, so a count gold ("3") can never be read off
    an object phrase. Falls back to ordinal() from 100 up, which no released
    cell reaches.
    """
    if n < 1:
        raise ValueError(f"ordinal_word needs a positive integer, got {n}")
    if n < 20:
        return _ORDINAL_WORDS[n]
    if n < 100:
        tens, units = divmod(n, 10)
        if units == 0:
            return _TENS_ORDINAL[tens]
        return f"{_TENS_CARDINAL[tens]}-{_ORDINAL_WORDS[units]}"
    return ordinal(n)


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
        ordered = sorted(containers)
        k = len(ordered)
        if k > min(len(CONTAINER_ADJS), len(CONTAINER_NOUNS)):
            raise ValueError(
                f"{k} containers exceed the container vocabulary "
                f"({len(CONTAINER_ADJS)} adjectives, {len(CONTAINER_NOUNS)} nouns)"
            )
        # Adjectives and nouns are each drawn without replacement, so no two
        # containers in an instance share either word: "the red box" and "the
        # red chest" would let a partial match ("the red ...") pass as correct
        # and make the reader disambiguate on a single token.
        adjs = rng.sample(CONTAINER_ADJS, k)
        nouns = rng.sample(CONTAINER_NOUNS, k)
        for cid, adj, noun in zip(ordered, adjs, nouns):
            self.container_names[cid] = f"the {adj} {noun}"

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

        Same-type objects are named by order of first appearance ("the first
        key", "the second key", ...), whether they came from independent Puts or
        a Split: "original"/"duplicate" implied a copy relation that only Split
        creates. Addressable without a WorldState, so a Put can name the object
        it introduces before the state contains it (render_narrative passes the
        creation rank it derived from the trace).
        """
        return f"the {ordinal_word(rank)} {obj_type}"


# Distractor templates (SPEC §2, N). Each takes the plural of an object type
# that is NOT used in the instance, so a distractor never names a tracked type,
# a container or a quantity; that is why the answer-leakage gate cannot fire on
# them. Wording rules, checked by test/test_render_names_distractors.py: no
# digits or number words, no container adjective/noun, and no object type as a
# substring of any template word ("opened" hides "pen", "during" hides "ring").
DISTRACTOR_TEMPLATES: Tuple[str, ...] = (
    "Someone mentioned that {plural} have become harder to find lately.",
    "A neighbour said that {plural} were on sale at the market.",
    "There was a brief conversation about {plural} over breakfast.",
    "Somebody remarked that {plural} are often overlooked.",
    "A friend recalled reading an article about {plural}.",
    "The radio played a short segment about {plural}.",
    "Someone wondered aloud where {plural} are usually sold.",
    "A visitor asked whether {plural} were still in fashion.",
    "It was said that {plural} used to be made by hand.",
    "A child drew a picture of {plural} that afternoon.",
    "Someone joked that {plural} are more trouble than they are worth.",
    "A shopkeeper complained that {plural} had grown costly.",
    "A familiar song mentioned {plural} in its chorus.",
    "Someone read a short story about {plural} aloud.",
    "A guest asked where good {plural} could be bought.",
    "People in town seemed to talk about {plural} a lot.",
)

# Type-free scene sentences, used only to top up the pool when the instance
# uses so many object types that the template x unused-type product is smaller
# than the requested N. Same wording rules as DISTRACTOR_TEMPLATES.
DISTRACTOR_FLAVOR: Tuple[str, ...] = (
    "The afternoon light was fading.",
    "Outside, it had started to drizzle.",
    "A dog barked somewhere down the street.",
    "The kettle whistled in the kitchen.",
    "Someone hummed a tune while walking past.",
    "The clock in the hall chimed softly.",
    "A breeze rattled the window panes.",
    "The neighbours were talking in the garden.",
    "Somebody laughed in the next room.",
    "A car drove slowly past the house.",
    "The smell of fresh bread drifted in.",
    "Rain tapped lightly against the roof.",
    "A bird settled on the fence outside.",
    "The radio was playing quietly.",
    "Someone sighed and stretched.",
    "The evening was calm and quiet.",
)


def make_distractor_sentences(
    rng: random.Random,
    n: int,
    names: NameRegistry,
    used_object_types: Sequence[str],
) -> List[str]:
    """Return ``n`` pairwise-distinct distractor sentences.

    Candidates are the product DISTRACTOR_TEMPLATES x (OBJECT_TYPES minus the
    instance's types), in fixed vocabulary order, topped up with
    DISTRACTOR_FLAVOR only when that product is smaller than ``n``; ``n`` are
    drawn without replacement with ``rng``. No candidate names a container, a
    count or a type used in the instance (see the template comment).
    ``names`` is accepted for the call contract; the sentences need no names.
    """
    if n <= 0:
        return []
    used = set(used_object_types)
    # Fixed vocabulary order, never set order (hard rule 5).
    unrelated = [t for t in OBJECT_TYPES if t not in used]
    pool = [
        template.format(plural=pluralize_object(t))
        for template in DISTRACTOR_TEMPLATES
        for t in unrelated
    ]
    if len(pool) < n:
        pool.extend(DISTRACTOR_FLAVOR)
    if len(pool) < n:
        raise ValueError(
            f"cannot draw {n} distinct distractor sentences: only {len(pool)} "
            f"candidates for used types {sorted(used)}"
        )
    return rng.sample(pool, n)


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
