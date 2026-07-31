#!/usr/bin/env python3

"""
Accuracy and Agreement Metrics

Evaluates each model and the weighted-mean ensemble against human grades on
the 80% test split, per university and pooled. Also reports dataset info
and intra-model (AI vs AI) agreement.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.scoring_utils import ScoringUtils
from src.utils.db_utils import get_connection
from src.utils.calibration_utils import get_all_calibration_ids
from src.utils.analysis_utils import (
    INSTITUTIONS,
    ANALYSIS_METHODS,
    load_institution_ticks,
    load_complete_case_data,
    compute_accuracy_metrics,
    compute_pooled_metrics,
    round_floats,
    print_metrics_table,
)


METHOD_LABELS = {
    "gemini": "Gemini",
    "gpt": "GPT",
    "claude": "Claude",
    "wmean": "Weighted Mean Ensemble",
}
BAND_ORDER = ["First", "Upper Second", "Lower Second", "Third", "Fail"]

# Decimal places per JSON key ("_pct" is a suffix match)
PRECISION = {
    "_pct": 1, "mae": 2, "rmse": 2,
    "qwk": 3, "pearson": 3, "spearman": 3,
    "icc_2_k": 3, "krippendorff_alpha": 3,
    "grade_mean": 2, "grade_std": 2,
}


# Dataset statistics

def dataset_info(human: np.ndarray) -> Dict:
    """Distribution summary for a set of human grades."""
    bands = [ScoringUtils.get_grade_band(int(g)) for g in human]
    band_counts = {b: int(bands.count(b)) for b in BAND_ORDER}
    return {
        "n": int(len(human)),
        "grade_mean": float(human.mean()),
        "grade_std": float(human.std(ddof=1)) if len(human) > 1 else 0.0,
        "grade_min": int(human.min()),
        "grade_max": int(human.max()),
        "band_counts": band_counts,
    }


# Intra-model agreement

def icc_2_k(matrix: np.ndarray) -> float:
    """ICC(2,k) two-way random, average measures, absolute agreement."""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_row = k * np.sum((row_means - grand) ** 2)
    ss_col = n * np.sum((col_means - grand) ** 2)
    ss_tot = np.sum((matrix - grand) ** 2)
    ss_err = ss_tot - ss_row - ss_col
    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_row + (ms_col - ms_err) / n
    return float((ms_row - ms_err) / denom) if denom else float("nan")


def krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    """Krippendorff's α (interval) — rows = essays, cols = raters, complete."""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    d_obs_sum = 0.0
    pairs = 0
    for row in matrix:
        for a, b in itertools.combinations(row, 2):
            d_obs_sum += (a - b) ** 2
            pairs += 1
    d_obs = d_obs_sum / pairs
    flat = matrix.flatten()
    d_exp = np.mean([(a - b) ** 2 for a, b in itertools.combinations(flat, 2)])
    return float(1 - d_obs / d_exp) if d_exp else float("nan")


def pairwise_agreement(a: np.ndarray, b: np.ndarray) -> Dict:
    """Agreement metrics between two AI raters on the same essays."""
    bands_a = np.array([ScoringUtils.get_grade_band(int(x)) for x in a])
    bands_b = np.array([ScoringUtils.get_grade_band(int(x)) for x in b])
    return {
        "n": int(len(a)),
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "qwk": float(cohen_kappa_score(
            a.astype(int), b.astype(int), weights="quadratic"
        )),
        "exact_band_pct": float(np.mean(bands_a == bands_b) * 100),
    }


def intra_model_block(data_for_uni: Dict[str, np.ndarray]) -> Dict:
    """Intra-model (AI vs AI) agreement for one dataset slice."""
    ai_models = ("gemini", "gpt", "claude")
    matrix = np.column_stack([data_for_uni[m] for m in ai_models])
    block = {
        "n_essays": int(matrix.shape[0]),
        "icc_2_k": icc_2_k(matrix),
        "krippendorff_alpha": krippendorff_alpha_interval(matrix),
        "pairs": {},
    }
    for a, b in itertools.combinations(ai_models, 2):
        block["pairs"][f"{a}_vs_{b}"] = pairwise_agreement(data_for_uni[a], data_for_uni[b])
    return block


def print_intra_model_table(title: str, block: Dict) -> None:
    """Print the intra-model agreement summary for one dataset slice."""
    print(f"\n  {title} — Intra-model (AI vs AI) agreement  (n={block['n_essays']})")
    print(f"  ICC(2,k) = {block['icc_2_k']:.3f}   "
          f"Krippendorff α = {block['krippendorff_alpha']:.3f}")
    print(f"  {'Pair':<22}{'n':>5}{'r':>8}{'ρ':>8}{'QWK':>8}{'Band%':>8}")
    for pair, s in block["pairs"].items():
        a, b = pair.split("_vs_")
        label = f"{a.capitalize()} vs {b.capitalize()}"
        print(f"  {label:<22}{s['n']:>5}{s['pearson']:>8.3f}{s['spearman']:>8.3f}"
              f"{s['qwk']:>8.3f}{s['exact_band_pct']:>7.1f}%")


# Main

def main():
    parser = argparse.ArgumentParser(description="Accuracy and agreement metrics.")
    parser.add_argument(
        "--no-test-split", action="store_true",
        help="Include calibration essays (default: exclude, use 80%% test split)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "output" / "results.json",
        help="Path for the results JSON",
    )
    parser.add_argument("--db-path", type=Path, default=None)
    args = parser.parse_args()

    conn = get_connection(args.db_path)
    exclude_ids = set() if args.no_test_split else get_all_calibration_ids(conn)[0]
    data = load_complete_case_data(conn, exclude_ids=exclude_ids)
    conn.close()

    if not data:
        print("No data loaded.")
        sys.exit(1)

    ticks_map = load_institution_ticks()
    split_label = ("full dataset" if args.no_test_split
                   else f"80% test split (excluded {len(exclude_ids)} calibration essays)")

    print(f"{'=' * 110}")
    print(f"  MAIN STATISTICS — {split_label}")
    print(f"{'=' * 110}")

    output: Dict = {
        "split": "full" if args.no_test_split else "test_80",
        "n_calibration_excluded": len(exclude_ids),
        "methods": list(ANALYSIS_METHODS),
        "universities": {},
        "pooled": {},
    }

    for uni in [v for v in INSTITUTIONS.values() if v in data]:
        human = data[uni]["human"]
        info = dataset_info(human)
        results = {
            m: compute_accuracy_metrics(human, data[uni][m], ticks_map[uni])
            for m in ANALYSIS_METHODS
        }
        intra = intra_model_block(data[uni])

        print_metrics_table(
            uni.upper(), [(METHOD_LABELS[m], results[m]) for m in ANALYSIS_METHODS],
            info=info,
        )
        print_intra_model_table(uni, intra)

        output["universities"][uni] = {
            "dataset": info,
            "accuracy": results,
            "intra_model_agreement": intra,
        }

    pooled_human = np.concatenate([data[u]["human"] for u in data])
    pooled_info = dataset_info(pooled_human)
    pooled_results = {
        m: compute_pooled_metrics(
            {u: data[u]["human"] for u in data},
            {u: data[u][m] for u in data},
            ticks_map,
        )
        for m in ANALYSIS_METHODS
    }
    pooled_intra = intra_model_block({
        m: np.concatenate([data[u][m] for u in data])
        for m in ("gemini", "gpt", "claude")
    })

    print_metrics_table(
        "POOLED (ALL UNIVERSITIES)",
        [(METHOD_LABELS[m], pooled_results[m]) for m in ANALYSIS_METHODS],
        info=pooled_info,
    )
    print_intra_model_table("Pooled", pooled_intra)

    output["pooled"] = {
        "dataset": pooled_info,
        "accuracy": pooled_results,
        "intra_model_agreement": pooled_intra,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(round_floats(output, PRECISION), f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
