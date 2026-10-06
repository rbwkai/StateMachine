"""
eval/prompts.py
================
Unified prompt construction module for DWS-Bench.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Answer-slot text per query type (SPEC.md §6: prompts live only in this module).
# Slot text is inserted after ``str.format`` (via sentinels) so it is never
# parsed as format fields. Slots carry no braces: an unfilled ``{type}`` would
# reach the model verbatim.
ANSWER_SLOTS: Dict[str, Dict[str, str]] = {
    "location": {
        "final": "Final Answer: <name of the container>",
        "step": "name of the container the object is in after that event",
    },
    "count": {
        "final": "Final Answer: <a single integer>",
        "step": "number of objects of the asked type in the asked container after that event",
    },
}

QUERY_TYPES: Tuple[str, ...] = tuple(ANSWER_SLOTS)

# Sentinels so slot text can carry literal braces without being consumed as
# ``str.format`` fields.
_FINAL_SLOT_TOKEN = "\x00final-slot\x00"
_STEP_SLOT_TOKEN = "\x00step-slot\x00"


V1_NON_COT_TEMPLATE = (
    "Narrative:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "{final_slot}\n"
    "Stop immediately after Final Answer."
)

V1_COT_TEMPLATE = (
    "Narrative:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Solve the problem by updating the world state step by step.\n\n"
    "Step 1: {step_slot}\n"
    "Step 2: {step_slot}\n"
    "Step 3: {step_slot}\n\n"
    "{final_slot}\n"
    "Stop immediately after Final Answer."
)

V2_NON_COT_TEMPLATE = (
    "Narrative:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer the question using only the narrative above.\n"
    "{final_slot}\n"
    "Stop immediately after Final Answer."
)

V2_COT_TEMPLATE = (
    "Narrative:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Solve the problem by updating the world state step by step.\n"
    "For each event in the narrative, write one line in order as Step k: <{step_slot}>, using as many steps as there are events.\n"
    "Then write a final line: {final_slot}\n"
    "Stop immediately after Final Answer."
)


def build_user_prompt(
    context: str,
    question: str,
    chain_of_thought: bool = False,
    prompt_version: str = "v2",
    query_type: str = "location",
) -> str:
    """
    Build user prompt string for state tracking tasks.

    Args:
        context: The narrative context string.
        question: The question string.
        chain_of_thought: Whether to include step-by-step reasoning instructions.
        prompt_version: Prompt specification version ("v1" or "v2").
        query_type: Answer contract of the question ("location" or "count").
    """
    if prompt_version == "v1":
        template = V1_COT_TEMPLATE if chain_of_thought else V1_NON_COT_TEMPLATE
    elif prompt_version == "v2":
        template = V2_COT_TEMPLATE if chain_of_thought else V2_NON_COT_TEMPLATE
    else:
        raise ValueError(f"Unknown prompt_version: '{prompt_version}'. Must be 'v1' or 'v2'.")

    slots = ANSWER_SLOTS.get(query_type)
    if slots is None:
        raise ValueError(
            f"Unknown query_type: '{query_type}'. Must be one of {list(QUERY_TYPES)}."
        )

    return (
        template.format(
            context=context,
            question=question,
            final_slot=_FINAL_SLOT_TOKEN,
            step_slot=_STEP_SLOT_TOKEN,
        )
        .replace(_FINAL_SLOT_TOKEN, slots["final"])
        .replace(_STEP_SLOT_TOKEN, slots["step"])
    )

# ============================================================
# Robustness prompt variants (E4). Owned here with every other prompt string
# (AGENTS.md §5); eval/robustness.py imports and re-exports them.
# ============================================================

PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "v1_original": {
        "system": "You are a helpful assistant that answers questions about object locations.",
        "instruction": "Read the following narrative and answer the question.",
    },
    "v2_standard": {
        "system": "You are an expert dynamic state reasoning assistant. Answer the question based only on the given narrative state changes. Give your final answer clearly.",
        "instruction": "Track the object locations through the narrative and answer the question.",
    },
    "v3_minimal": {
        "system": "Answer the question based on the narrative.",
        # Placeholder: the minimal instruction depends on the query type and is
        # taken from MINIMAL_INSTRUCTIONS in build_prompt_variants().
        "instruction": "",
    },
}

# The minimal variant restates the question kind. A fixed "Where is the
# object?" contradicted the integer answer slot of count records, so the
# variant measured a prompt conflict instead of template sensitivity. Keys
# must equal QUERY_TYPES (the ANSWER_SLOTS keys).
MINIMAL_INSTRUCTIONS: Dict[str, str] = {
    "location": "Where is the object?",
    "count": "How many objects are in the container?",
}
if set(MINIMAL_INSTRUCTIONS) != set(QUERY_TYPES):
    raise AssertionError(
        f"MINIMAL_INSTRUCTIONS keys {sorted(MINIMAL_INSTRUCTIONS)} != "
        f"QUERY_TYPES {sorted(QUERY_TYPES)}"
    )


def _variant_instruction(variant: str, query_type: str) -> str:
    if variant == "v3_minimal":
        if query_type not in MINIMAL_INSTRUCTIONS:
            raise ValueError(
                f"Unknown query_type: '{query_type}'. "
                f"Must be one of {sorted(MINIMAL_INSTRUCTIONS)}."
            )
        return MINIMAL_INSTRUCTIONS[query_type]
    return PROMPT_TEMPLATES[variant]["instruction"]


def build_prompt_variants(
    context: str,
    question: str,
    chain_of_thought: bool = False,
    query_type: str = "location",
) -> Dict[str, str]:
    """Build all prompt template variants for a given instance.

    Returns a dict mapping variant name to the full prompt text including the
    variant's system prompt, embedded inline so mock and real engines see the
    same single user message.
    """
    variants: Dict[str, str] = {}
    for name, template in PROMPT_TEMPLATES.items():
        instruction = _variant_instruction(name, query_type)
        user_prompt = build_user_prompt(
            context=f"{instruction}\n\n{context}",
            question=question,
            chain_of_thought=chain_of_thought,
            prompt_version="v2",
            query_type=query_type,
        )
        variants[name] = f"[SYSTEM_PROMPT: {template['system']}]\n{user_prompt}"
    return variants
