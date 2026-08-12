"""Minimal end-to-end demo: generate a handful of examples and print them
in a readable form. Run with: python3 example.py
"""
import json
import random

from pipeline import generate_example
from world import Put, Move, Remove, Undo, Redo, Split, Merge, Swap

ALL_OPS = [Put, Move, Remove, Undo, Redo, Split, Merge, Swap]


def print_example(ex: dict) -> None:
    print(f"--- {ex['id']}  (factors: {ex['factors']}) ---")
    for s in ex["sentences"]:
        print(f"  {s}")
    print(f"Q: {ex['question']}")
    print(f"A: {ex['gold_answer']}")
    if "step_wise_gold" in ex:
        print(f"Step-wise gold: {ex['step_wise_gold']}")
    if ex.get("counterfactual_probes"):
        print("Counterfactual probes:")
        for p in ex["counterfactual_probes"]:
            print(f"  - {p['question']}  ->  {p['gold_answer']}")
    print()


if __name__ == "__main__":
    rng = random.Random(0)

    # A forced redo-validity probe.
    ex3 = generate_example(
        rng, "redo_validity_example", entity_count=4, update_count=7,
        distractor_count=1, operations_enabled=ALL_OPS, force_redo_probe=True,
    )
    print_example(ex3)