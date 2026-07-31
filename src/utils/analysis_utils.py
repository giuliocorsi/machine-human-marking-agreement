#!/usr/bin/env python3

"""
Analysis utilities for essay scoring.
Shared data loading, accuracy metrics, and output formatting for the analysis scripts.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from .scoring_utils import ScoringUtils

PROJECT_ROOT = Path(__file__).parent.parent.parent

# University code → database display name
INSTITUTIONS = {"cambridge": "Cambridge", "mmu": "MMU", "nottingham": "Nottingham"}

# The three models plus the weighted-mean ensemble
ANALYSIS_METHODS = ("gemini", "gpt", "claude", "wmean")


# Data loading for analysis scripts

def load_institution_ticks() -> Dict[str, List[int]]:
    """Load permitted tick positions per university from the institution configs."""
    ticks_map = {}
    for code, name in INSTITUTIONS.items():
        config_path = PROJECT_ROOT / "src" / "config" / "institutions" / f"{code}.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        ticks_map[name] = sorted(int(v) for v in config["classification"]["possible_values"])
    return ticks_map


def load_complete_case_data(
    conn: sqlite3.Connection,
    exclude_ids: Optional[set] = None,
    include_ids: Optional[set] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load paired human/AI grades per university for the complete-case corpus
    (long essays with a human grade and scores from all models + ensemble).

    Returns:
        {university: {"human": array, "gemini": array, "gpt": array,
                      "claude": array, "wmean": array}}
    """
    rows = conn.execute("""
        SELECT id, university, human_grade,
               gemini_score, gpt_score, claude_score, wmean_score
        FROM essays
        WHERE LOWER(essay_type) = 'long'
          AND human_grade IS NOT NULL
          AND gemini_score IS NOT NULL
          AND gpt_score    IS NOT NULL
          AND claude_score IS NOT NULL
          AND wmean_score  IS NOT NULL
        ORDER BY id
    """).fetchall()

    data: Dict[str, Dict[str, list]] = {}
    for r in rows:
        if exclude_ids and r["id"] in exclude_ids:
            continue
        if include_ids is not None and r["id"] not in include_ids:
            continue
        bucket = data.setdefault(
            r["university"], {"human": [], **{m: [] for m in ANALYSIS_METHODS}}
        )
        bucket["human"].append(r["human_grade"])
        for m in ANALYSIS_METHODS:
            bucket[m].append(r[f"{m}_score"])

    return {
        uni: {k: np.array(v, dtype=float) for k, v in d.items()}
        for uni, d in data.items()
    }


# Accuracy metrics

def compute_accuracy_metrics(
    human: np.ndarray, preds: np.ndarray, ticks: List[int]
) -> Dict[str, float]:
    """Accuracy / agreement metrics for one (human, preds) pair."""
    errors = preds - human
    tick_dists = np.array([
        ScoringUtils.tick_distance(h, p, ticks) for h, p in zip(human, preds)
    ])
    human_bands = np.array([ScoringUtils.get_grade_band(int(g)) for g in human])
    pred_bands = np.array([ScoringUtils.get_grade_band(int(p)) for p in preds])
    return {
        "n": int(len(human)),
        "exact_tick_pct": float(np.mean(tick_dists == 0) * 100),
        "within_1_tick_pct": float(np.mean(tick_dists <= 1) * 100),
        "within_2_tick_pct": float(np.mean(tick_dists <= 2) * 100),
        "band_accuracy_pct": float(np.mean(human_bands == pred_bands) * 100),
        "qwk": float(cohen_kappa_score(
            human.astype(int), preds.astype(int), weights="quadratic"
        )),
        "pearson": float(pearsonr(human, preds).statistic),
        "spearman": float(spearmanr(human, preds).statistic),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def compute_pooled_metrics(
    per_uni_human: Dict[str, np.ndarray],
    per_uni_preds: Dict[str, np.ndarray],
    ticks_map: Dict[str, List[int]],
) -> Dict[str, float]:
    """Pooled accuracy metrics — tick distances use each university's own scale."""
    all_tick_dists: List[int] = []
    all_err: List[float] = []
    all_human: List[int] = []
    all_preds: List[int] = []
    for uni in per_uni_human:
        u_human = per_uni_human[uni]
        u_preds = per_uni_preds[uni]
        u_ticks = ticks_map[uni]
        all_tick_dists.extend(
            ScoringUtils.tick_distance(h, p, u_ticks) for h, p in zip(u_human, u_preds)
        )
        all_err.extend((u_preds - u_human).tolist())
        all_human.extend(u_human.astype(int).tolist())
        all_preds.extend(u_preds.astype(int).tolist())
    tick_dists = np.array(all_tick_dists)
    errors = np.array(all_err)
    human = np.array(all_human)
    preds = np.array(all_preds)
    human_bands = np.array([ScoringUtils.get_grade_band(int(g)) for g in human])
    pred_bands = np.array([ScoringUtils.get_grade_band(int(g)) for g in preds])
    return {
        "n": int(len(tick_dists)),
        "exact_tick_pct": float(np.mean(tick_dists == 0) * 100),
        "within_1_tick_pct": float(np.mean(tick_dists <= 1) * 100),
        "within_2_tick_pct": float(np.mean(tick_dists <= 2) * 100),
        "band_accuracy_pct": float(np.mean(human_bands == pred_bands) * 100),
        "qwk": float(cohen_kappa_score(human, preds, weights="quadratic")),
        "pearson": float(pearsonr(human, preds).statistic),
        "spearman": float(spearmanr(human, preds).statistic),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


# Output formatting

def round_floats(d: Dict, precision: Optional[Dict[str, int]] = None, default: int = 3) -> Dict:
    """Recursively round all float values in a dict for readable JSON.
    precision maps key → decimal places; keys starting with "_" are suffix matches."""
    precision = precision or {}

    def round_value(key: str, value):
        if not isinstance(value, float):
            return value
        if key in precision:
            return round(value, precision[key])
        for suffix, dp in precision.items():
            if suffix.startswith("_") and key.endswith(suffix):
                return round(value, dp)
        return round(value, default)

    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = round_floats(v, precision, default)
        elif isinstance(v, list):
            out[k] = [round_floats(x, precision, default) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = round_value(k, v)
    return out


def print_metrics_table(
    title: str, rows: List[Tuple[str, Dict]], info: Optional[Dict] = None,
) -> None:
    """Print a standard accuracy-metrics table: one row per (label, metrics)."""
    print(f"\n{'-' * 110}")
    if info:
        print(f"  {title}  (n={info['n']}, mean={info['grade_mean']:.1f}, "
              f"std={info['grade_std']:.1f}, range=[{info['grade_min']}, {info['grade_max']}])")
    else:
        print(f"  {title}")
    print(f"{'-' * 110}")
    print(f"  {'Method':<26}{'n':>5}{'Exact':>8}{'±1T':>7}{'±2T':>7}{'Band':>7}"
          f"{'QWK':>7}{'r':>7}{'ρ':>7}{'MAE':>7}{'RMSE':>7}")
    print(f"  {'-' * 108}")
    for label, m in rows:
        print(
            f"  {label:<26}{m['n']:>5}"
            f"{m['exact_tick_pct']:>7.1f}%"
            f"{m['within_1_tick_pct']:>6.1f}%"
            f"{m['within_2_tick_pct']:>6.1f}%"
            f"{m['band_accuracy_pct']:>6.1f}%"
            f"{m['qwk']:>7.3f}"
            f"{m['pearson']:>7.3f}"
            f"{m['spearman']:>7.3f}"
            f"{m['mae']:>7.2f}"
            f"{m['rmse']:>7.2f}"
        )
