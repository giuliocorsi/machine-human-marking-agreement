"""
Prompt Renderer - Renders prompt templates with institution-specific variables.

Usage:
    from prompts.prompt_renderer import PromptRenderer

    renderer = PromptRenderer()
    prompt = renderer.render("A2_B2_C1", "cambridge")
    all_prompts = renderer.render_all("mmu")
"""

import json
from pathlib import Path
from typing import Optional


class PromptRenderer:
    """Renders prompt templates with institution-specific variables."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "prompt_library.json"

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.institutions = self.config["institutions"]
        self.templates = {t["id"]: t for t in self.config["prompt_templates"]}
        self.components = self.config["components"]
        # A library may delegate the output contract to build_output_prompt, which
        # builds the JSON schema from the institution's permitted marks.
        self.output_format = self.config.get("output_format", "")

    def get_grade_bands(self, institution: str) -> str:
        """Get grade band names for an institution."""
        terms = self.institutions[institution]["grade_terminology"]
        return f"{terms['fail']}/{terms['third']}/{terms['lower_second']}/{terms['upper_second']}/{terms['first']}"

    def render(self, prompt_id: str, institution: str) -> str:
        """
        Render a specific prompt template for an institution.

        Args:
            prompt_id: The prompt ID (e.g., "A2_B2_C1")
            institution: The institution key (e.g., "cambridge", "mmu", "nottingham")

        Returns:
            The fully rendered prompt string.
        """
        if prompt_id not in self.templates:
            raise ValueError(f"Unknown prompt ID: {prompt_id}. Available: {list(self.templates.keys())}")

        if institution not in self.institutions:
            raise ValueError(f"Unknown institution: {institution}. Available: {list(self.institutions.keys())}")

        template = self.templates[prompt_id]["template"]
        inst = self.institutions[institution]
        grade_bands = self.get_grade_bands(institution)

        # Build replacement map from institution data
        replacements = {
            "{{BASE}}": self.components["base"]
                .replace("{{INSTITUTION}}", inst["name"])
                .replace("{{SUBJECT}}", inst["subject"]),
            "{{SUMMARY_RUBRIC}}": inst.get("summary_rubric", ""),
            "{{FULL_RUBRIC}}": inst.get("full_rubric", ""),
            "{{DISTRIBUTION}}": inst.get("distribution", ""),
            "{{OUTPUT_FORMAT}}": self.output_format,
        }

        # Add all components dynamically, substituting nested placeholders
        for key, value in self.components.items():
            if key == "base":
                continue  # already handled above with institution substitution
            placeholder = "{{" + key.upper() + "}}"
            value = value.replace("{{GRADE_BANDS}}", grade_bands)
            value = value.replace("{{CRITERIA_LIST}}", inst.get("criteria_list", ""))
            replacements[placeholder] = value

        # Apply replacements
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def render_all(self, institution: str) -> dict[str, str]:
        """
        Render all prompt templates for an institution.

        Args:
            institution: The institution key

        Returns:
            Dictionary mapping prompt IDs to rendered prompts.
        """
        return {
            prompt_id: self.render(prompt_id, institution)
            for prompt_id in self.templates.keys()
        }

    def get_prompt_info(self, prompt_id: str) -> dict:
        """Get metadata about a prompt template."""
        if prompt_id not in self.templates:
            raise ValueError(f"Unknown prompt ID: {prompt_id}")
        return {
            "id": self.templates[prompt_id]["id"],
            "name": self.templates[prompt_id]["name"],
            "description": self.templates[prompt_id]["description"],
        }

    def list_prompts(self) -> list[dict]:
        """List all available prompt templates with metadata."""
        return [
            {
                "id": t["id"],
                "name": t["name"],
                "description": t["description"],
            }
            for t in self.config["prompt_templates"]
        ]

    def list_institutions(self) -> list[str]:
        """List all available institutions."""
        return list(self.institutions.keys())


def main():
    """Demo the prompt renderer."""
    renderer = PromptRenderer()

    print("Available institutions:", renderer.list_institutions())
    print("\nAvailable prompts (27 total):")
    for p in renderer.list_prompts():
        print(f"  {p['id']}: {p['name']}")

    print("\n" + "="*80)
    print("Example: A2_B2_C2 for Cambridge")
    print("="*80)
    print(renderer.render("A2_B2_C2", "cambridge"))


if __name__ == "__main__":
    main()
