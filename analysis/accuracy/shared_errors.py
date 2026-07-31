#!/usr/bin/env python3

"""
Shared-Error Analysis

Do the three models' errors covary? Pairwise signed-error correlations
(raw and partialling out the human mark), plus the proportion of essays
where all three models err in the same direction vs an independence benchmark.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db_utils import get_connection
from src.utils.calibration_utils import get_all_calibration_ids
from src.utils.analysis_utils import INSTITUTIONS, load_complete_case_data


MODELS = ("claude", "gemini", "gpt")


def residualize(values: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Residuals of a linear regression of `values` on `on`."""
    slope, intercept = np.polyfit(on, values, 1)
    return values - (slope * on + intercept)


def error_stats(errors: Dict[str, np.ndarray], human: np.ndarray) -> Dict:
    """Pairwise error correlations and same-direction proportions for one slice."""
    n = len(next(iter(errors.values())))
    resid = {m: residualize(errors[m], human) for m in MODELS}
    pair_r = {}
    for a, b in combinations(MODELS, 2):
        r, p = pearsonr(errors[a], errors[b])
        r_partial, _ = pearsonr(resid[a], resid[b])
        pair_r[f"{a}-{b}"] = {"r": float(r), "p": float(p),
                              "r_partial_human": float(r_partial)}

    signs = np.stack([np.sign(errors[m]) for m in MODELS])   # 3 × n, values −1/0/+1
    all_neg = np.all(signs < 0, axis=0)
    all_pos = np.all(signs > 0, axis=0)
    same_dir = float(np.mean(all_neg | all_pos))

    # Independence benchmark from each model's marginal sign frequencies
    p_neg = [float(np.mean(errors[m] < 0)) for m in MODELS]
    p_pos = [float(np.mean(errors[m] > 0)) for m in MODELS]
    expected = float(np.prod(p_neg) + np.prod(p_pos))

    return {
        "n": n,
        "pairwise_error_r": pair_r,
        "mean_pairwise_r": float(np.mean([v["r"] for v in pair_r.values()])),
        "mean_pairwise_r_partial_human": float(
            np.mean([v["r_partial_human"] for v in pair_r.values()])),
        "same_direction": {
            "observed": same_dir,
            "all_below": float(np.mean(all_neg)),
            "all_above": float(np.mean(all_pos)),
            "expected_if_independent": expected,
            "ratio": same_dir / expected if expected > 0 else float("nan"),
        },
    }


def print_stats(scope: str, stats: Dict) -> None:
    """Print the error-covariation summary for one scope."""
    sd = stats["same_direction"]
    print(f"{scope} (n = {stats['n']})")
    for pair, v in stats["pairwise_error_r"].items():
        print(f"  r(err {pair}) = {v['r']:.3f}   "
              f"partial (| human) = {v['r_partial_human']:.3f}")
    print(f"  mean pairwise r = {stats['mean_pairwise_r']:.3f}   "
          f"partial = {stats['mean_pairwise_r_partial_human']:.3f}")
    print(f"  same direction: {sd['observed']*100:.1f}% "
          f"(all below {sd['all_below']*100:.1f}%, all above {sd['all_above']*100:.1f}%; "
          f"independent expectation {sd['expected_if_independent']*100:.1f}%, "
          f"ratio {sd['ratio']:.2f})\n")


def main():
    parser = argparse.ArgumentParser(description="Shared-error analysis across models.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "output" / "shared_errors.json",
                        help="Path for the results JSON")
    args = parser.parse_args()

    conn = get_connection()
    exclude_ids, _ = get_all_calibration_ids(conn)
    data = load_complete_case_data(conn, exclude_ids=exclude_ids)
    conn.close()

    universities = list(INSTITUTIONS.values())
    pooled_human = np.concatenate([data[u]["human"] for u in universities])
    pooled_errors = {
        m: np.concatenate([data[u][m] - data[u]["human"] for u in universities])
        for m in MODELS
    }

    results: Dict = {
        "pooled": error_stats(pooled_errors, pooled_human),
        "institutions": {},
    }
    for uni in universities:
        errors = {m: data[uni][m] - data[uni]["human"] for m in MODELS}
        results["institutions"][uni] = error_stats(errors, data[uni]["human"])

    print_stats("POOLED", results["pooled"])
    for uni in universities:
        print_stats(uni, results["institutions"][uni])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
