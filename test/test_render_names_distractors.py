"""
Render-layer naming and distractor properties (render/names.py).

- distractor sentences: pairwise distinct up to N=16, no container name or
  container word, no digit or number word, no object type used in the instance,
  same bytes for the same rng;
- container names: no adjective and no noun shared within an instance;
- same-type objects: named by a stable ordinal of first appearance, and Split
  narrates which ordinal is the copy.
"""
from __future__ import annotations

import itertools
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest  # noqa: E402

from world import Move, Put, Split, WorldState  # noqa: E402
from render.names import (  # noqa: E402
    CONTAINER_ADJS,
    CONTAINER_NOUNS,
    DISTRACTOR_FLAVOR,
    DISTRACTOR_TEMPLATES,
    OBJECT_TYPES,
    NameRegistry,
    make_distractor_sentences,
    ordinal_word,
    pluralize_object,
)
from render.narrative import render_narrative  # noqa: E402
from test.conftest import build_record, quiet  # noqa: E402

MAX_N = 16  # largest textual-distractor level (experiments/rq2_interference.py)
NUMBER_WORDS = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"dozen|several|few|both|pair|couple|single|once|twice|removed)\b"
)
CONTAINER_WORDS = set(CONTAINER_ADJS) | set(CONTAINER_NOUNS)

# Used-type sets covering the fallback boundaries: none used, a typical
# instance, all but one type (template product == 16 == MAX_N), and all types
# (template product empty, flavor only).
USED_TYPE_SETS = [
    [],
    ["key", "gem"],
    ["phone", "pen", "map", "ring", "watch"],
    list(OBJECT_TYPES[:-1]),
    list(OBJECT_TYPES),
]


def _assert_clean(sentence, used_types, container_names):
    low = sentence.lower()
    assert not re.search(r"\d", sentence), sentence
    assert not NUMBER_WORDS.search(low), sentence
    assert not (set(re.findall(r"[a-z]+", low)) & CONTAINER_WORDS), sentence
    for name in container_names:
        assert name not in low, sentence
    for t in used_types:
        # Substring, not whole word: the record-level leakage check in
        # test_rendering.py matches tracked types as substrings.
        assert t not in low and pluralize_object(t) not in low, (t, sentence)


# ---------------------------------------------------------------------------
# distractors
# ---------------------------------------------------------------------------

def test_vocabulary_supports_sixteen_distinct_distractors():
    assert len(DISTRACTOR_TEMPLATES) >= 12
    assert len(set(DISTRACTOR_TEMPLATES)) == len(DISTRACTOR_TEMPLATES)
    assert len(set(DISTRACTOR_FLAVOR)) == len(DISTRACTOR_FLAVOR) >= MAX_N
    for template in DISTRACTOR_TEMPLATES:
        assert template.count("{plural}") == 1, template


def test_template_wording_hides_no_object_type_or_container_word():
    """No template or flavor word contains any object type as a substring."""
    for text in DISTRACTOR_TEMPLATES + DISTRACTOR_FLAVOR:
        _assert_clean(text.replace("{plural}", ""), OBJECT_TYPES, [])


@pytest.mark.parametrize("used", USED_TYPE_SETS, ids=lambda u: f"used{len(u)}")
@pytest.mark.parametrize("n", [1, 4, 8, MAX_N])
def test_distractors_are_distinct_and_leak_nothing(used, n):
    containers = {f"c{i}" for i in range(4)}
    for seed in range(30):
        names = NameRegistry(random.Random(seed), containers)
        container_names = set(names.container_names.values())
        sentences = make_distractor_sentences(random.Random(seed), n, names, used)
        assert len(sentences) == n
        assert len(set(sentences)) == n, sentences
        for sentence in sentences:
            _assert_clean(sentence, used, container_names)


@pytest.mark.parametrize("used", USED_TYPE_SETS, ids=lambda u: f"used{len(u)}")
def test_distractors_are_deterministic_for_a_fixed_rng(used):
    names = NameRegistry(random.Random(0), {"c0", "c1", "c2"})
    for seed in range(10):
        a = make_distractor_sentences(random.Random(seed), MAX_N, names, used)
        b = make_distractor_sentences(random.Random(seed), MAX_N, names, list(reversed(used)))
        assert a == b


def test_distractors_vary_across_seeds():
    names = NameRegistry(random.Random(0), {"c0", "c1", "c2"})
    draws = {
        tuple(make_distractor_sentences(random.Random(seed), MAX_N, names, ["key"]))
        for seed in range(20)
    }
    assert len(draws) == 20


def test_zero_distractors_consume_no_randomness():
    names = NameRegistry(random.Random(0), {"c0", "c1"})
    rng = random.Random(5)
    state = rng.getstate()
    assert make_distractor_sentences(rng, 0, names, ["key"]) == []
    assert rng.getstate() == state


def test_more_distractors_than_candidates_is_rejected():
    names = NameRegistry(random.Random(0), {"c0", "c1"})
    with pytest.raises(ValueError):
        make_distractor_sentences(
            random.Random(0), len(DISTRACTOR_FLAVOR) + 1, names, list(OBJECT_TYPES)
        )


def test_record_with_sixteen_distractors_has_sixteen_distinct_ones():
    """End to end at the RQ2 cell shape (interleaved_chain, E=3, 4 containers)."""
    pool = {
        t.format(plural=pluralize_object(o))
        for t, o in itertools.product(DISTRACTOR_TEMPLATES, OBJECT_TYPES)
    } | set(DISTRACTOR_FLAVOR)
    checked = 0
    for seed in range(5):
        with quiet():
            record = build_record(
                "interleaved_chain", seed, entity_count=3, target_updates=8,
                distractor_updates=4, num_containers=4, textual_distractors=MAX_N,
                max_attempts=20,
            )
        if record is None:
            continue
        checked += 1
        distractors = [s for s in record["sentences"] if s in pool]
        assert len(distractors) == MAX_N, record["sentences"]
        assert len(set(distractors)) == MAX_N, distractors
        used = {s["obj_type"] for s in record["canonical_trace"] if s.get("obj_type")}
        names = set(record["final_state"]["container_display_names"].values())
        for sentence in distractors:
            _assert_clean(sentence, used, names)
    assert checked, "no interleaved_chain record built at N=16"


# ---------------------------------------------------------------------------
# container names
# ---------------------------------------------------------------------------

MAX_CONTAINERS = min(len(CONTAINER_ADJS), len(CONTAINER_NOUNS))


@pytest.mark.parametrize("k", range(2, MAX_CONTAINERS + 1))
def test_container_names_share_no_adjective_and_no_noun(k):
    containers = {f"c{i}" for i in range(k)}
    for seed in range(100):
        names = NameRegistry(random.Random(seed), containers)
        parts = [n.split(" ") for n in names.container_names.values()]
        assert all(len(p) == 3 and p[0] == "the" for p in parts), parts
        adjs = [p[1] for p in parts]
        nouns = [p[2] for p in parts]
        assert len(set(adjs)) == k, adjs
        assert len(set(nouns)) == k, nouns


def test_container_names_are_deterministic_for_a_fixed_rng():
    containers = {f"c{i}" for i in range(6)}
    for seed in range(20):
        a = NameRegistry(random.Random(seed), containers).container_names
        b = NameRegistry(random.Random(seed), set(sorted(containers, reverse=True))).container_names
        assert a == b


def test_too_many_containers_for_the_vocabulary_is_rejected():
    with pytest.raises(ValueError):
        NameRegistry(random.Random(0), {f"c{i}" for i in range(MAX_CONTAINERS + 1)})


# ---------------------------------------------------------------------------
# same-type object naming
# ---------------------------------------------------------------------------

def test_ordinal_words():
    expected = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 9: "ninth",
        11: "eleventh", 12: "twelfth", 19: "nineteenth", 20: "twentieth",
        21: "twenty-first", 32: "thirty-second", 43: "forty-third",
        90: "ninetieth", 99: "ninety-ninth",
    }
    for n, word in expected.items():
        assert ordinal_word(n) == word
    with pytest.raises(ValueError):
        ordinal_word(0)


def test_same_type_names_are_ordinal_by_first_appearance_and_stable():
    containers = {"c0", "c1", "c2"}
    names = NameRegistry(random.Random(0), containers)
    ops = [
        Put("o0", "key", "c0"),
        Put("o1", "coin", "c1"),
        Put("o2", "key", "c1"),
        Move("o0", "c2"),
        Put("o3", "key", "c0"),
        Move("o2", "c0"),
        Move("o3", "c2"),
        Move("o1", "c0"),
    ]
    sentences, state = render_narrative(ops, containers, names)
    text = " ".join(sentences)
    assert "original" not in text and "duplicate" not in text
    assert re.search(r"\b\d+(st|nd|rd|th)\b", text) is None, text
    assert sentences[0].endswith("as the first key."), sentences
    assert sentences[1].startswith("A coin was placed in "), sentences
    assert sentences[1].endswith(f"{names.container('c1')}."), sentences
    assert sentences[2].endswith("as the second key."), sentences
    assert sentences[3].startswith("The first key was moved"), sentences
    assert sentences[4].endswith("as the third key."), sentences
    assert sentences[5].startswith("The second key was moved"), sentences
    assert sentences[6].startswith("The third key was moved"), sentences
    assert sentences[7].startswith("The coin was moved"), sentences
    # The final-state names (what the question uses) match the narrative.
    assert [names.obj(o, state) for o in ("o0", "o2", "o3", "o1")] == [
        "the first key", "the second key", "the third key", "the coin",
    ]


def test_split_names_the_copy_and_the_source_by_their_ordinals():
    from eval.robustness import apply_paraphrase

    containers = {"c0", "c1", "c2"}
    names = NameRegistry(random.Random(0), containers)
    ops = [Put("o0", "key", "c0"), Split("o0", "o1"), Move("o1", "c2"), Move("o0", "c1")]
    sentences, state = render_narrative(ops, containers, names)
    c0 = names.container("c0")
    assert sentences[0] == f"A key was placed in {c0} as the first key."
    assert sentences[1] == (
        f"The second key was placed in {c0} as an identical copy of the first key."
    )
    assert sentences[2].startswith("The second key was moved"), sentences
    assert sentences[3].startswith("The first key was moved"), sentences
    assert names.obj("o1", state) == "the second key"
    # The split sentence keeps the "<word> was placed in <phrase>." shape.
    assert apply_paraphrase(sentences[1]) != sentences[1]
