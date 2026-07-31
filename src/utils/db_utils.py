#!/usr/bin/env python3

"""
Database utilities for the OPrAISE essay classifier.
Works with the root opraise.db — a single flat essays table with all predictions inline.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional

from .scoring_utils import ScoringUtils

# Default database path — root opraise.db, overridable via OPRAISE_DB env var
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DB_PATH = Path(os.getenv("OPRAISE_DB", str(PROJECT_ROOT / "data" / "opraise.db")))

# Prediction column pairs: prefix → (score_col, band_col)
PREDICTION_COLUMNS = {
    "gemini":   ("gemini_score",   "gemini_band"),
    "gpt":      ("gpt_score",      "gpt_band"),
    "claude":   ("claude_score",   "claude_band"),
    "median":   ("median_score",   "median_band"),
    "mean":     ("mean_score",     "mean_band"),
    "surp_pop": ("surp_pop_score", "surp_pop_band"),
    "wmean":    ("wmean_score",    "wmean_band"),
}

# Feedback columns: model prefix → feedback_col
FEEDBACK_COLUMNS = {
    "gemini": "gemini_feedback",
    "gpt":    "gpt_feedback",
    "claude": "claude_feedback",
}

# Map aggregator names (as returned by the classifier) to column prefixes
_AGG_NAME_MAP = {
    "median_vote": "median",
    "simple_mean": "mean",
    "surprisingly_popular": "surp_pop",
    "weighted_mean": "wmean",
}

# Map model display names (case-insensitive) to column prefixes
_MODEL_NAME_MAP = {
    "gemini": "gemini",
    "gpt": "gpt",
    "claude": "claude",
}

# Re-export for convenience
extract_grade = ScoringUtils.extract_grade


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection with row factory enabled."""
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=DELETE")
    return conn


def ensure_prediction_columns(conn: sqlite3.Connection) -> None:
    """Add prediction columns to the essays table if they don't exist yet."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(essays)").fetchall()}
    for prefix, (score_col, band_col) in PREDICTION_COLUMNS.items():
        if score_col not in existing:
            conn.execute(f"ALTER TABLE essays ADD COLUMN {score_col} INTEGER")
        if band_col not in existing:
            conn.execute(f"ALTER TABLE essays ADD COLUMN {band_col} TEXT")
        # Add feedback column right after band for each model
        if prefix in FEEDBACK_COLUMNS:
            feedback_col = FEEDBACK_COLUMNS[prefix]
            if feedback_col not in existing:
                conn.execute(f"ALTER TABLE essays ADD COLUMN {feedback_col} TEXT")
    conn.commit()


def ensure_calibration_table(
    conn: sqlite3.Connection, table: str = "calibration_scores"
) -> None:
    """Create a calibration scores table if it doesn't exist.

    Stores one row per (essay, model, prompt) combination scored during
    the factorial calibration sweep. Each prompt library version writes to
    its own table (declared via ``calibration_table`` in the library JSON)
    so sweeps under different libraries never overwrite each other.
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            essay_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            score INTEGER,
            band TEXT,
            raw_response TEXT,
            UNIQUE(essay_id, model, prompt_id),
            FOREIGN KEY (essay_id) REFERENCES essays(id)
        )
    """)
    conn.commit()


def save_predictions(
    conn: sqlite3.Connection,
    essay_db_id: int,
    model_predictions: Dict[str, Dict[str, Any]],
    aggregated_results: Dict[str, Dict[str, Any]],
) -> None:
    """Write model predictions and aggregated results for a single essay.

    Updates the flat columns on the essays table directly.
    """
    flat_updates: Dict[str, Any] = {}

    # --- Model predictions ---
    for model_name, pred in model_predictions.items():
        classification = extract_grade(pred.get("classification"))
        grade_band = pred.get("grade_band")
        feedback = pred.get("feedback")

        prefix = _MODEL_NAME_MAP.get(model_name.lower())
        if prefix:
            score_col, band_col = PREDICTION_COLUMNS[prefix]
            flat_updates[score_col] = classification
            flat_updates[band_col] = grade_band
            if prefix in FEEDBACK_COLUMNS and feedback:
                flat_updates[FEEDBACK_COLUMNS[prefix]] = feedback

    # --- Aggregated results ---
    for agg_name, agg in aggregated_results.items():
        if not isinstance(agg, dict):
            continue
        classification = extract_grade(agg.get("classification"))
        grade_band = agg.get("grade_band")

        prefix = _AGG_NAME_MAP.get(agg_name)
        if prefix:
            score_col, band_col = PREDICTION_COLUMNS[prefix]
            flat_updates[score_col] = classification
            flat_updates[band_col] = grade_band

    # --- Update flat columns on essays table ---
    if flat_updates:
        set_clause = ", ".join(f"{col} = ?" for col in flat_updates)
        values = list(flat_updates.values()) + [essay_db_id]
        conn.execute(f"UPDATE essays SET {set_clause} WHERE id = ?", values)
        conn.commit()


def get_institution_data(
    conn: sqlite3.Connection,
    institution_name: Optional[str] = None,
    exclude_ids: Optional[set] = None,
    essay_type: Optional[str] = None,
    complete_cases: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Load essay data with predictions for one or all universities.

    Returns dict keyed by university name, each containing:
        human_grades: list of all human grades
        model_preds:  {name: {"truths": [...], "preds": [...]}}
        aggregated_preds: {name: {"truths": [...], "preds": [...]}}

    Each model/aggregator carries its own paired truth/pred lists,
    so only essays with predictions for that method are included.

    Args:
        exclude_ids: Optional set of essay row IDs to exclude (e.g. calibration set).
        essay_type: Optional filter for essay type (e.g. "long", "short").
        complete_cases: If True, keep only essays with a human grade and a
            non-null score from all three models and the weighted-mean
            ensemble, so every method is evaluated on the identical essay set
            (the paper's complete-case corpus).
    """
    clauses = []
    params: List = []
    if institution_name:
        clauses.append("university = ?")
        params.append(institution_name)
    if essay_type:
        clauses.append("LOWER(essay_type) = ?")
        params.append(essay_type.lower())
    if complete_cases:
        clauses.append(
            "human_grade IS NOT NULL AND gemini_score IS NOT NULL "
            "AND gpt_score IS NOT NULL AND claude_score IS NOT NULL "
            "AND wmean_score IS NOT NULL"
        )
    query = "SELECT * FROM essays"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY university, id"

    rows = conn.execute(query, params).fetchall()

    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if exclude_ids and row["id"] in exclude_ids:
            continue
        uni = row["university"]
        if uni not in result:
            result[uni] = {"human_grades": [], "model_preds": {}, "aggregated_preds": {}}

        human_grade = row["human_grade"]
        if human_grade is None:
            continue
        result[uni]["human_grades"].append(human_grade)

        # Read model predictions from flat columns — paired with human grade
        for prefix, (score_col, _band_col) in PREDICTION_COLUMNS.items():
            score = row[score_col] if score_col in row.keys() else None
            if score is None:
                continue

            if prefix in ("median", "mean", "surp_pop", "wmean"):
                section = result[uni]["aggregated_preds"]
            else:
                section = result[uni]["model_preds"]

            if prefix not in section:
                section[prefix] = {"truths": [], "preds": []}
            section[prefix]["truths"].append(human_grade)
            section[prefix]["preds"].append(score)

    return result
