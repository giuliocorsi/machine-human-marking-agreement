"""Prompt manager — appends structured JSON output format to rendered prompts.

The existing ``PromptRenderer`` (in ``prompts/prompt_renderer.py``) handles
template rendering with institution-specific variables.  This module adds the
JSON output-format instructions and surprisingly-popular prediction request
that the LLM needs to return structured results.
"""

import json
from typing import Dict, Any, List, Optional


def build_output_prompt(
    rendered_prompt: str,
    task_config: Dict[str, Any],
    include_explanation: bool = False,
    include_feedback: bool = False,
) -> str:
    """Append JSON output-format instructions to a rendered prompt.

    Args:
        rendered_prompt: The prompt text already rendered by PromptRenderer.
        task_config: The per-university task YAML (must contain
            ``classification.possible_values``).
        include_explanation: Whether to ask for an explanation field.

    Returns:
        The full prompt string ready to send to an LLM.
    """
    possible_values = _get_possible_values(task_config)

    # Build the predicted_distribution schema
    predicted_distribution = {v: "number: between 0 and 100" for v in possible_values}

    values_str = ", ".join(possible_values)
    surp_text = (
        f"\n    Additionally, predict how other experts would classify this text:"
        f"\n    - What percentage would classify it as {values_str}?"
    )

    output_format: Dict[str, Any] = {
        "classification": f"string: one of {possible_values}",
        "confidence": "number: between 0 and 100",
        "predicted_distribution": predicted_distribution,
    }
    if include_feedback:
        output_format["feedback"] = "string: feedback explaining the mark in under 200 words"
    if include_explanation:
        output_format["explanation"] = "string: detailed explanation"

    return (
        f"{rendered_prompt}{surp_text}\n\n"
        f"IMPORTANT: Respond with ONLY a valid JSON object containing these fields:\n"
        f"{json.dumps(output_format, indent=2)}\n\n"
        f"The response must be a valid JSON object without any text outside it."
    )


def build_model_prompt(
    renderer,
    prompt_id: str,
    institution: str,
    task_config: Dict[str, Any],
    include_explanation: bool = False,
    include_feedback: bool = False,
) -> str:
    """Render a prompt, appending the JSON output block unless the library supplies its own."""
    rendered = renderer.render(prompt_id, institution)
    if renderer.config.get("self_contained_output", False):
        return rendered
    return build_output_prompt(
        rendered, task_config,
        include_explanation=include_explanation, include_feedback=include_feedback,
    )


def _get_possible_values(task_config: Dict[str, Any]) -> List[str]:
    """Extract possible classification values from task config."""
    try:
        return task_config["classification"]["possible_values"]
    except (KeyError, TypeError):
        raise ValueError(
            "task_config must contain classification.possible_values"
        )
