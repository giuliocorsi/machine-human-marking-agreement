"""Calibration utilities — random split logic for calibration/test partitioning.

The split is fully deterministic given (seed, data). No database flags needed.
Essays are sampled randomly within each university to ensure proportional
institutional representation while preserving the natural grade distribution.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent


def random_split(
    rows: List[Dict[str, Any]],
    fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into calibration and test sets by random sampling.

    Uses numpy for reproducible random sampling. At least one essay is
    assigned to calibration.

    Returns:
        (calibration_rows, test_rows)
    """
    rng = np.random.default_rng(seed)
    n_cal = max(1, round(len(rows) * fraction))
    indices = rng.permutation(len(rows))

    cal_rows = [rows[idx] for idx in indices[:n_cal]]
    test_rows = [rows[idx] for idx in indices[n_cal:]]

    return cal_rows, test_rows


def get_calibration_split(
    conn: sqlite3.Connection,
    university: str,
    seed: int = 42,
    fraction: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load essays for a university and return (calibration, test) split.

    Args:
        conn: Database connection.
        university: Institution name (case-insensitive).
        seed: Random seed for reproducible split.
        fraction: Proportion of essays assigned to calibration (~0.2).

    Returns:
        (calibration_rows, test_rows) — each a list of dicts with
        id, student_id, human_grade, human_grade_band, assignment_brief, assignment_content.
    """
    rows = conn.execute(
        "SELECT id, student_id, human_grade, human_grade_band, assignment_brief, assignment_content "
        "FROM essays WHERE LOWER(university) = ? AND LOWER(essay_type) = 'long' ORDER BY id",
        (university.lower(),),
    ).fetchall()

    if not rows:
        return [], []

    row_dicts = [dict(r) for r in rows]
    return random_split(row_dicts, fraction=fraction, seed=seed)


def get_all_calibration_ids(
    conn: sqlite3.Connection,
    universities: Tuple[str, ...] = ("cambridge", "mmu", "nottingham"),
) -> Tuple[set, Dict[str, Dict[str, Any]]]:
    """Collect the calibration-split essay IDs across all universities,
    using each institution's YAML seed/fraction.

    Returns:
        (calibration_ids, splits_info) — pooled ID set, plus per-university
        {"seed", "fraction", "n_cal"} keyed by university code.
    """
    cal_ids: set = set()
    splits_info: Dict[str, Dict[str, Any]] = {}
    for university in universities:
        config_path = PROJECT_ROOT / "src" / "config" / "institutions" / f"{university}.yaml"
        if not config_path.exists():
            continue
        with open(config_path) as f:
            config = yaml.safe_load(f)
        seed = config.get("calibration", {}).get("seed", 42)
        fraction = config.get("calibration", {}).get("fraction", 0.2)
        cal_rows, _ = get_calibration_split(conn, university, seed=seed, fraction=fraction)
        ids = {r["id"] for r in cal_rows}
        cal_ids |= ids
        splits_info[university] = {"seed": seed, "fraction": fraction, "n_cal": len(ids)}
    return cal_ids, splits_info
