"""Run-condition resolution for saved predictions (model, CoT mode, token budget).

Kept free of eval/generator imports so plotting can resolve conditions without
loading the scorer (AGENTS.md §4); `evaluate_existing_predictions` and
`plot_results` both import it, so there is one resolver (§5).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

# run_eval.py writes <output_dir>/<model>/<cot|no_cot>_<tokens>/<dataset>_predictions.jsonl
# and, since the layout fix, the same fields on every prediction row. Older
# exports used one <model>_<cot|no_cot>_<tokens> directory two levels up.
_RUN_DIR_RE = re.compile(r"^(?P<cot>no_cot|cot)_(?P<tokens>\d{1,7})$")
_LEGACY_DIR_RE = re.compile(r"^(?P<model>.+?)_(?P<cot>no_cot|cot)_(?P<tokens>\d{1,7})$")
# Model names reach CSV cells and plot labels: a registry-style name only, so
# a crafted row cannot inject a spreadsheet formula ("=...") or mathtext ("$").
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d{1,7}", value.strip()):
        return int(value.strip())
    return None


def _condition_from_path(path: Path) -> Dict[str, Any]:
    match = _RUN_DIR_RE.match(path.parent.name)
    if match and path.parent.parent.name:
        return {
            "model": path.parent.parent.name,
            "cot": match.group("cot") == "cot",
            "max_new_tokens": int(match.group("tokens")),
        }
    match = _LEGACY_DIR_RE.match(path.parent.parent.name)
    if match:
        return {
            "model": match.group("model"),
            "cot": match.group("cot") == "cot",
            "max_new_tokens": int(match.group("tokens")),
        }
    return {}


def infer_condition(prediction_path: Path, row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve (model, cot, max_new_tokens) for one prediction row.

    Each field comes from the row (``model``, ``chain_of_thought``,
    ``max_new_tokens``) when present, else from the path: first the run_eval
    layout ``<model>/<cot|no_cot>_<tokens>/<file>``, then the legacy
    ``<model>_<cot|no_cot>_<tokens>/<subdir>/<file>``. Raises ValueError when a
    field cannot be resolved: a silent default would re-score CoT runs without
    Step-protocol enforcement.
    """
    row = row or {}
    from_path = _condition_from_path(prediction_path)

    from_row = {
        "model": str(row["model"]) if row.get("model") else None,
        "cot": _as_bool(row.get("chain_of_thought")),
        "max_new_tokens": _as_int(row.get("max_new_tokens")),
    }
    # Row and path both say something: they must agree, or a row could turn a
    # CoT run into a no-CoT re-score and skip Step-protocol enforcement.
    conflicts = [
        f"{name}: row={from_row[name]!r} path={from_path[name]!r}"
        for name in from_row
        if from_row[name] is not None and from_path.get(name) is not None
        and from_row[name] != from_path[name]
    ]
    if conflicts:
        raise ValueError(
            f"Run condition in {prediction_path} contradicts its directory "
            f"layout: {'; '.join(conflicts)}"
        )
    model = from_row["model"] if from_row["model"] is not None else from_path.get("model")
    cot = from_row["cot"] if from_row["cot"] is not None else from_path.get("cot")
    tokens = (from_row["max_new_tokens"] if from_row["max_new_tokens"] is not None
              else from_path.get("max_new_tokens"))
    if model is not None and not _MODEL_NAME_RE.match(model):
        raise ValueError(f"Unsafe model name {model!r} in {prediction_path}")

    missing = [
        name for name, value in
        (("model", model), ("chain_of_thought", cot), ("max_new_tokens", tokens))
        if value is None
    ]
    if missing:
        raise ValueError(
            f"Cannot resolve run condition {missing} for {prediction_path}: the row "
            f"does not carry them and the path matches neither "
            f"'<model>/<cot|no_cot>_<tokens>/<file>' nor "
            f"'<model>_<cot|no_cot>_<tokens>/<subdir>/<file>'."
        )
    mode = "cot" if cot else "no_cot"
    return {
        "model": model,
        "cot": cot,
        "max_new_tokens": tokens,
        "condition": f"{model}_{mode}_{tokens}",
    }
