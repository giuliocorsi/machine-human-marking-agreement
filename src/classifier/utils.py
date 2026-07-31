"""Small utility functions used across the classifier package."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_input_data(path: str) -> Dict[str, Any]:
    """Load a JSON data file."""
    with open(path, "r") as f:
        return json.load(f)


def save_data(path: str, data: Any) -> None:
    """Write *data* as pretty-printed JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_base_filename(path: str) -> str:
    """Return the stem (filename without extension) of *path*."""
    return Path(path).stem
