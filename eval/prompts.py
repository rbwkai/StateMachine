"""
eval/prompts.py
================
Unified prompt construction module for DWS-Bench.
"""

from __future__ import annotations

from typing import Optional


V1_NON_COT_TEMPLATE = (
    "Narrative:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Final Answer: <answer>\n"
    "Stop immediately after Final Answer."
)

V1_COT_TEMPLATE = (
    "Narrative:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Solve the problem by updating the world state step by step.\n\n"
    "Step 1: <container>\n"
    "Step 2: <container>\n"
    "Step 3: <container>\n\n"
    "Final Answer: <answer>\n"
    "Stop immediately after Final Answer."
)

V2_NON_COT_TEMPLATE = (
    "Narrative:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Final Answer: <name of the container>\n"
    "Stop immediately after Final Answer."
)

V2_COT_TEMPLATE = (
    "Narrative:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Solve the problem by updating the world state step by step.\n"
    "For each event in the narrative, write one line in order as Step k: <name of the container the object is in after that event>, using as many steps as there are events.\n"
    "Then write a final line: Final Answer: <name of the container>\n"
    "Stop immediately after Final Answer."
)


def build_user_prompt(
    context: str,
    question: str,
    chain_of_thought: bool = False,
    prompt_version: str = "v2",
) -> str:
    """
    Build user prompt string for state tracking tasks.

    Args:
        context: The narrative context string.
        question: The question string.
        chain_of_thought: Whether to include step-by-step reasoning instructions.
        prompt_version: Prompt specification version ("v1" or "v2").
    """
    if prompt_version == "v1":
        template = V1_COT_TEMPLATE if chain_of_thought else V1_NON_COT_TEMPLATE
    elif prompt_version == "v2":
        template = V2_COT_TEMPLATE if chain_of_thought else V2_NON_COT_TEMPLATE
    else:
        raise ValueError(f"Unknown prompt_version: '{prompt_version}'. Must be 'v1' or 'v2'.")

    return template.format(context=context, question=question)
