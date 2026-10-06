"""
Checklist 4 — Rendering: sentence counts, names, distractors, paraphrase, text hygiene.

Everything here reads a generated record (one replay pass upstream) plus the
render layer's own helpers. Tests named ``test_failsnow_*`` encode the CORRECT
property for a defect recorded in the pre-freeze audit.
"""
from __future__ import annotations

import collections
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest  # noqa: E402

from world import Put, Move  # noqa: E402
from render.names import (  # noqa: E402
    CONTAINER_ADJS,
    CONTAINER_NOUNS,
    OBJECT_TYPES,
    NameRegistry,
    make_distractor_sentences,
    pluralize_object,
    splice_distractors,
)
from test.conftest import build_record, quiet  # noqa: E402

SEEDS = range(20)
UNDO_SENTENCE = "That last action was undone."

RECORD_CELLS = [
    ("basic_chain", 1, 4, 0),
    ("interleaved_chain", 2, 4, 2),
    ("revision", 1, 4, 0),
    ("split_chain", 2, 4, 0),
    ("undo_chain", 1, 4, 0),
    ("undo_redo_chain", 1, 8, 0),
]
CELL_IDS = [f"{f}-T{t}" for f, _e, t, _d in RECORD_CELLS]
DISTRACTOR_COUNTS = [0, 4, 8, 16]


def records_for(family, entity_count, target_updates, distractor_updates=0, n_textual=0, seeds=SEEDS):
    with quiet():
        out = [
            build_record(
                family, seed,
                entity_count=entity_count,
                target_updates=target_updates,
                distractor_updates=distractor_updates,
                textual_distractors=n_textual,
                max_attempts=20,
            )
            for seed in seeds
        ]
    return [r for r in out if r is not None]


def op_sentence_indices(record):
    """Indices of the sentences that came from operations, in order."""
    return list(range(len(record["canonical_trace"])))


def distractor_sentences(record):
    """Sentences that are not in the op-sentence prefix order.

    The record keeps sentences in narrative order; the op count is known, so a
    distractor is any sentence whose text does not appear among the sentences
    rendered for the trace ops. Cheap and exact for these families because every
    op sentence names either an object type or a container.
    """
    ids = set(record["final_state"]["container_display_names"].values())
    types = {s.get("obj_type") for s in record["canonical_trace"] if s.get("obj_type")}
    out = []
    for sentence in record["sentences"]:
        names_container = any(name in sentence for name in ids)
        names_type = any(t in sentence for t in types)
        if not names_container and not names_type:
            out.append(sentence)
    return out


# ---------------------------------------------------------------------------
# sentence count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,e,t,d", RECORD_CELLS, ids=CELL_IDS)
@pytest.mark.parametrize("n_textual", DISTRACTOR_COUNTS)
def test_sentence_count_equals_op_count_plus_n(family, e, t, d, n_textual):
    records = records_for(family, e, t, d, n_textual)
    assert records, f"no instances for {family} T={t} N={n_textual}"
    for record in records:
        assert len(record["sentences"]) == len(record["canonical_trace"]) + n_textual, (
            f"{record['instance_id']}: {len(record['sentences'])} sentences for "
            f"{len(record['canonical_trace'])} ops and N={n_textual}"
        )


@pytest.mark.parametrize("family,e,t,d", RECORD_CELLS, ids=CELL_IDS)
def test_question_names_the_target_like_the_narrative(family, e, t, d):
    records = records_for(family, e, t, d)
    assert records, f"no instances for {family} T={t}"
    for record in records:
        question = record["question"]
        if question.startswith("Where is "):
            phrase = question[len("Where is "):-len(" now?")].strip()
        elif question.startswith("How many "):
            phrase = question[len("How many "):].split(" are in ")[0]
        else:
            continue
        # Compare content words: the narrative says "A pen"/"The pen" and
        # "letter" where the question says "the duplicate key"/"letters".
        content = re.sub(r"^(a|an|the)\s+", "", phrase.lower()).strip()
        candidates = {content}
        if content.endswith("es"):
            candidates.add(content[:-2])
        if content.endswith("s"):
            candidates.add(content[:-1])
        narrative = " ".join(record["sentences"]).lower()
        assert any(c in narrative for c in candidates), (
            f"{record['instance_id']}: question phrase {phrase!r} never appears in "
            f"the narrative"
        )
        if question.startswith("How many "):
            container = question.split(" are in ")[-1].removesuffix(" now?").strip().lower()
            assert container in narrative, (
                f"{record['instance_id']}: question container {container!r} never "
                f"appears in the narrative"
            )


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------

def test_ordinals_run_through_the_twelfth_object():
    """NameRegistry.obj() must produce original, duplicate and 3rd..12th."""
    from world import WorldState

    containers = {"c0"}
    state = WorldState(
        object_type={f"o{i}": "key" for i in range(12)},
        location={f"o{i}": "c0" for i in range(12)},
        containers=containers,
    )
    names = NameRegistry(random.Random(0), containers)
    rendered = [names.obj(f"o{i}", state) for i in range(12)]
    assert len(set(rendered)) == 12, rendered
    assert set(rendered) == {"the original key", "the duplicate key"} | {
        f"the {n}{suffix} key"
        for n, suffix in zip(range(3, 13), ["rd", "th", "th", "th", "th", "th", "th",
                                             "th", "th", "th"])
    }, sorted(rendered)


def test_ordinal_suffixes_follow_the_english_rules():
    """11th/12th/13th take 'th'; 21st/22nd/23rd do not."""
    from world import WorldState

    containers = {"c0"}
    names = NameRegistry(random.Random(0), containers)
    state = WorldState(
        object_type={f"o{i}": "key" for i in range(23)},
        location={f"o{i}": "c0" for i in range(23)},
        containers=containers,
    )
    rendered = {names.obj(f"o{i}", state) for i in range(23)}
    for expected in ("the 11th key", "the 12th key", "the 13th key", "the 20th key",
                     "the 21st key", "the 22nd key", "the 23rd key"):
        assert expected in rendered, sorted(rendered)


def test_failsnow_ordinals_follow_creation_order_not_lexicographic_id_order():
    """Ordinals should follow the order the objects appear in the story.

    Currently same-type objects are ranked with sorted(obj_id), so 'o10' and
    'o11' are named 'the 5th key' and 'the 6th key' while 'o3' is 'the 3rd key'.
    The reader sees 'the 12th key' before 'the 3rd key' and has no way to know
    which object was introduced first. Not in the pre-freeze [fails now] list;
    found by this suite.
    """
    from world import WorldState

    containers = {"c0"}
    state = WorldState(
        object_type={f"o{i}": "key" for i in range(12)},
        location={f"o{i}": "c0" for i in range(12)},
        containers=containers,
    )
    names = NameRegistry(random.Random(0), containers)
    assert names.obj("o2", state) == "the 3rd key"
    assert names.obj("o11", state) == "the 12th key"


def test_failsnow_original_and_duplicate_are_explained_to_the_reader():
    """[checklist 4] "'original' and 'duplicate' are defined to the reader
    somewhere."

    The renderer switches to 'the original key' / 'the duplicate key' as soon as
    two objects share a type, but no sentence in the narrative explains which
    object each word refers to, so a reader cannot resolve the reference from the
    context alone. Not in the pre-freeze [fails now] list; found by this suite.
    """
    from render.narrative import render_narrative

    containers = {"c0", "c1", "c2"}
    names = NameRegistry(random.Random(0), containers)
    ops = [Put("o0", "key", "c0"), Put("o1", "key", "c1"), Move("o0", "c2")]
    sentences, _state = render_narrative(ops, containers, names)
    assert any("the original" in s or "the duplicate" in s for s in sentences)
    context = " ".join(sentences).lower()
    explains = any(
        word in context for word in ("first", "second", "copy", "copies", "split", "another")
    )
    assert explains, (
        "narrative uses original/duplicate without telling the reader which "
        f"object is which: {sentences}"
    )


def test_all_object_types_get_a_correct_article_and_plural():
    assert len(OBJECT_TYPES) == 13
    vowels = "aeiou"
    for obj_type in OBJECT_TYPES:
        plural = pluralize_object(obj_type)
        assert plural.endswith("s")
        assert obj_type in plural or obj_type[:-1] in plural, f"{obj_type} -> {plural}"
        for sentence in (f"A {obj_type} was placed in the red box.",
                         f"An {obj_type} was placed in the red box."):
            pass
    # a/an agreement is a renderer concern; check the helper it uses
    from render.narrative import _indefinite_article

    for obj_type in OBJECT_TYPES:
        article = _indefinite_article(obj_type).lower()
        expected = "an" if obj_type[0].lower() in vowels else "a"
        assert article == expected, f"{obj_type}: got {article!r}, expected {expected!r}"


@pytest.mark.parametrize("num_containers", [2, 3, 4])
def test_container_names_are_unique(num_containers):
    import random

    containers = {f"c{i}" for i in range(num_containers)}
    for seed in range(50):
        names = NameRegistry(random.Random(seed), containers)
        rendered = list(names.container_names.values())
        assert len(set(rendered)) == num_containers, f"duplicate container names: {rendered}"


@pytest.mark.parametrize("family,e,t,d", RECORD_CELLS, ids=CELL_IDS)
def test_container_display_names_are_unique_in_the_record(family, e, t, d):
    records = records_for(family, e, t, d)
    assert records
    for record in records:
        names = record["final_state"]["container_display_names"]
        assert len(set(names.values())) == len(names)


# ---------------------------------------------------------------------------
# distractors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_textual", DISTRACTOR_COUNTS)
def test_failsnow_distractors_never_leak_answers_objects_or_containers(n_textual):
    """[checklist 4] Distractors 'never contain the gold answer, a container
    name, or a tracked object's type.'

    currently: leakage rules hold for the released cells, but this test also
    guards the case where the *tracked type* is the only content of a sentence,
    so the check is kept as a property rather than a snapshot.
    """
    leaks = []
    for family, e, t, d in RECORD_CELLS:
        for record in records_for(family, e, t, d, n_textual, seeds=range(10)):
            types = {
                s["obj_type"] for s in record["canonical_trace"] if s.get("obj_type")
            }
            types |= {pluralize_object(t_) for t_ in types}
            container_names = set(record["final_state"]["container_display_names"].values())
            gold = record["gold_answer"]
            for sentence in distractor_sentences(record):
                if gold and gold in sentence:
                    leaks.append((record["instance_id"], "gold", sentence))
                if any(name in sentence for name in container_names):
                    leaks.append((record["instance_id"], "container", sentence))
                if any(obj in sentence for obj in types):
                    leaks.append((record["instance_id"], "tracked_type", sentence))
    assert not leaks, f"distractor leakage: {leaks[:6]}"


def test_failsnow_no_distractor_sits_between_an_op_and_the_undo_sentence():
    """[checklist 4] 'no distractor sits directly between an op and "That last
    action was undone."'

    [fails now] expected: the sentence before the Undo sentence is always the op
    sentence it inverts, so the reader never has to decide whether an inserted
    sentence was part of the story. currently: with N=8 an unrelated sentence
    lands directly in front of the Undo sentence in roughly half of the undo
    instances (25/40 in the audit run), which also breaks the narration order the
    renderer otherwise guarantees.
    """
    offenders = []
    checked = 0
    for record in records_for("undo_chain", 1, 8, 0, 8, seeds=range(40)):
        sentences = record["sentences"]
        try:
            undo_at = sentences.index(UNDO_SENTENCE)
        except ValueError:
            continue
        checked += 1
        previous = sentences[undo_at - 1] if undo_at else ""
        if previous in distractor_sentences(record):
            offenders.append(record["instance_id"])
    assert checked, "no undo_chain instance with N=8 to inspect"
    assert not offenders, (
        f"{len(offenders)}/{checked} undo instances have a distractor directly "
        f"before the Undo sentence: {offenders[:5]}"
    )


# ---------------------------------------------------------------------------
# splice_distractors
# ---------------------------------------------------------------------------

def test_splice_distractors_preserves_op_order():
    import random

    ops = [f"op{i}" for i in range(5)]
    extras = [f"d{i}" for i in range(3)]
    for seed in range(50):
        result = splice_distractors(random.Random(seed), ops, extras)
        assert len(result) == len(ops) + len(extras)
        assert [s for s in result if s in ops] == ops


def test_splice_distractors_handles_more_distractors_than_ops():
    import random

    ops = ["op0", "op1"]
    extras = [f"d{i}" for i in range(20)]
    result = splice_distractors(random.Random(0), ops, extras)
    assert len(result) == 22
    assert result[:2] == ops or [s for s in result if s in ops] == ops


def test_splice_distractors_with_no_distractors_returns_a_copy():
    import random

    ops = ["op0", "op1", "op2"]
    result = splice_distractors(random.Random(0), ops, [])
    assert result == ops
    assert result is not ops


def test_splice_positions_are_roughly_uniform():
    """Positions must not pile up at the start or the end."""
    import random

    ops = [f"op{i}" for i in range(4)]
    extras = [f"d{i}" for i in range(4)]
    slots = collections.Counter()
    trials = 2000
    for seed in range(trials):
        result = splice_distractors(random.Random(seed), ops, extras)
        slots[result.index("d0")] += 1
    # d0 is inserted when the sequence holds 4 items, so its final position is
    # uniform over 0..7 (4 ops + the 3 distractors inserted before it).
    assert set(slots) == set(range(len(ops) + len(extras)))
    expected = trials / len(slots)
    stat = sum((count - expected) ** 2 for count in slots.values()) / expected
    assert stat < 24.32, f"first distractor position is not uniform: {dict(slots)}"


def test_make_distractor_sentences_avoids_used_types_and_containers():
    import random

    containers = {"c0", "c1", "c2"}
    names = NameRegistry(random.Random(0), containers)
    container_names = set(names.container_names.values())
    for seed in range(50):
        sentences = make_distractor_sentences(
            random.Random(seed), 8, names, ["key", "gem"], exclude_container="c0"
        )
        assert len(sentences) == 8
        for sentence in sentences:
            assert "key" not in sentence and "gem" not in sentence and "keys" not in sentence
            for name in container_names:
                assert name not in sentence


# ---------------------------------------------------------------------------
# paraphrase
# ---------------------------------------------------------------------------

def test_every_sentence_type_has_a_paraphrase_rule():
    """[checklist 4] 'Paraphrase rules cover every sentence type (check split).'

    [fails now] expected: apply_paraphrase() changes every op sentence.
    currently: the Split sentence ('The letter in the wooden cabinet split into
    two identical copies.') has no rule, so the paraphrase condition E6 silently
    equals the baseline for every split_chain instance.
    """
    from eval.robustness import apply_paraphrase

    unchanged = collections.Counter()
    total = collections.Counter()
    for family, e, t, d in RECORD_CELLS:
        for record in records_for(family, e, t, d, seeds=range(10)):
            for op, sentence in zip(
                [s["op_type"] for s in record["canonical_trace"]], record["sentences"]
            ):
                total[op] += 1
                if apply_paraphrase(sentence) == sentence:
                    unchanged[op] += 1
    offenders = {op: f"{unchanged[op]}/{total[op]}" for op in total if unchanged[op]}
    assert not offenders, f"sentence types with no paraphrase rule: {offenders}"


@pytest.mark.parametrize("family,e,t,d", RECORD_CELLS, ids=CELL_IDS)
def test_paraphrase_never_changes_names_or_types(family, e, t, d):
    """Paraphrasing must not move the gold: names and object types survive."""
    from eval.robustness import apply_paraphrase

    records = records_for(family, e, t, d, seeds=range(10))
    assert records
    for record in records:
        paraphrased = [apply_paraphrase(s) for s in record["sentences"]]
        for name in record["final_state"]["container_display_names"].values():
            before = sum(name in s for s in record["sentences"])
            after = sum(name in s for s in paraphrased)
            assert before == after, (
                f"{record['instance_id']}: container {name!r} appears "
                f"{before}x before and {after}x after paraphrasing"
            )
        for obj_type in {s["obj_type"] for s in record["canonical_trace"] if s.get("obj_type")}:
            assert any(obj_type in s for s in paraphrased)


# ---------------------------------------------------------------------------
# text hygiene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,e,t,d", RECORD_CELLS, ids=CELL_IDS)
def test_context_is_a_plain_join_of_sentences(family, e, t, d):
    records = records_for(family, e, t, d, 4)
    assert records
    for record in records:
        assert record["context"] == " ".join(record["sentences"])
        assert "\n" not in record["context"] and "\t" not in record["context"]
        assert "  " not in record["context"]
        assert record["context"].strip() == record["context"]
        assert record["context"].isascii(), record["context"]
