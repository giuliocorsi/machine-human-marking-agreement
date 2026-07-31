#!/usr/bin/env python3

"""
Post-hoc Bias Corrections

Fits a per-model bias shift (pooled) and a per-institution affine rescaling
on the 20% calibration split, applies both to the 80% test set, and reports
before/after metrics with paired-bootstrap CIs on ΔMAE/ΔQWK.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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


MODEL_DISPLAY = {"gemini": "Gemini", "gpt": "GPT", "claude": "Claude", "wmean": "W-Mean"}
SCENARIOS = [("baseline", "Baseline"), ("corrected", "Bias Shift"), ("affine", "Affine")]

# Decimal places per JSON key ("_pct" is a suffix match)
PRECISION = {"_pct": 1}


# Corrections

def fit_bias(preds: np.ndarray, human: np.ndarray) -> float:
    """MLE of the translation parameter under Gaussian errors."""
    return float(np.mean(human - preds))


def fit_affine(preds: np.ndarray, human: np.ndarray) -> Tuple[float, float]:
    """Affine (a, b) matching the human mean and variance: a·pred + b."""
    a = float(np.std(human, ddof=1) / np.std(preds, ddof=1))
    b = float(np.mean(human) - a * np.mean(preds))
    return a, b


def snap_to_ticks(preds: np.ndarray, ticks: List[int]) -> np.ndarray:
    """Map corrected predictions to each institution's nearest permitted tick."""
    return np.array([ScoringUtils.find_nearest_tick(float(p), ticks) for p in preds],
                    dtype=float)


# Paired bootstrap

def paired_bootstrap_delta(
    human: np.ndarray, baseline: np.ndarray, corrected: np.ndarray,
    n_iter: int, seed: int,
) -> Dict:
    """95% paired-bootstrap CI on ΔMAE and ΔQWK (corrected − baseline)."""
    rng = np.random.default_rng(seed)
    n = len(human)
    base_err = np.abs(baseline - human)
    corr_err = np.abs(corrected - human)
    h_int = human.astype(int)
    b_int = baseline.astype(int)
    c_int = corrected.astype(int)

    mae_d = np.empty(n_iter)
    qwk_d = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        mae_d[i] = corr_err[idx].mean() - base_err[idx].mean()
        try:
            q_c = cohen_kappa_score(h_int[idx], c_int[idx], weights="quadratic")
            q_b = cohen_kappa_score(h_int[idx], b_int[idx], weights="quadratic")
            qwk_d[i] = q_c - q_b
        except Exception:
            qwk_d[i] = np.nan

    qwk_d = qwk_d[~np.isnan(qwk_d)]
    mae_ci = np.percentile(mae_d, [2.5, 97.5])
    qwk_ci = np.percentile(qwk_d, [2.5, 97.5]) if qwk_d.size else [np.nan, np.nan]
    return {
        "delta_mae_point": float(corr_err.mean() - base_err.mean()),
        "delta_qwk_point": float(
            cohen_kappa_score(h_int, c_int, weights="quadratic")
            - cohen_kappa_score(h_int, b_int, weights="quadratic")
        ),
        "delta_mae_mean": float(mae_d.mean()),
        "delta_mae_ci": [float(mae_ci[0]), float(mae_ci[1])],
        "mae_improves": bool(mae_ci[1] < 0),
        "delta_qwk_mean": float(qwk_d.mean()) if qwk_d.size else float("nan"),
        "delta_qwk_ci": [float(qwk_ci[0]), float(qwk_ci[1])],
        "qwk_improves": bool(qwk_ci[0] > 0) if qwk_d.size else False,
    }


# Display

def print_significance_block(results: Dict, key: str, title: str) -> None:
    """Print the paired-bootstrap significance table for one comparison."""
    print(f"\n{'=' * 110}")
    print(f"  SIGNIFICANCE — paired bootstrap 95% CI on {title} at the pooled test set")
    print("  ★ = CI excludes zero in the improving direction (ΔMAE<0 or ΔQWK>0).")
    print(f"{'=' * 110}")
    print(f"\n  {'Model':<12}{'ΔMAE':>10}{'CI(ΔMAE)':>20}{'  ':>4}"
          f"{'ΔQWK':>9}{'CI(ΔQWK)':>20}{'  ':>4}")
    print(f"  {'-' * 108}")
    for m in ANALYSIS_METHODS:
        s = results["per_model"][m][key]
        mae_mark = "★" if s["mae_improves"] else " "
        qwk_mark = "★" if s["qwk_improves"] else " "
        print(
            f"  {MODEL_DISPLAY[m]:<12}"
            f"{s['delta_mae_point']:>+10.3f}"
            f"  [{s['delta_mae_ci'][0]:+.2f}, {s['delta_mae_ci'][1]:+.2f}]{mae_mark:>3}"
            f"{s['delta_qwk_point']:>+9.3f}"
            f"  [{s['delta_qwk_ci'][0]:+.3f}, {s['delta_qwk_ci'][1]:+.3f}]{qwk_mark:>3}"
        )


def print_report(results: Dict) -> None:
    """Print the full before/after correction report."""
    print(f"\n{'=' * 110}")
    print("  POST-HOC CORRECTIONS — fitted on calibration split, applied to test set")
    print(f"{'=' * 110}")

    print(f"\n  Learned corrections per model (bias: pooled calibration, n = "
          f"{results['per_model']['gemini']['n_cal']}; affine: per institution)")
    print(f"  {'-' * 108}")
    for m in ANALYSIS_METHODS:
        bias = results["per_model"][m]["bias"]
        affine = ", ".join(
            f"{u} a={p['a']:.2f} b={p['b']:+.1f}"
            for u, p in results["per_model"][m]["affine_params"].items()
        )
        print(f"    {MODEL_DISPLAY[m]:<10}  bias = {bias:+.2f}   affine: {affine}")

    universities = list(results["per_model"]["gemini"]["per_university"].keys())
    for uni in universities:
        for scenario, scenario_label in SCENARIOS:
            rows = [
                (MODEL_DISPLAY[m], results["per_model"][m]["per_university"][uni][scenario])
                for m in ANALYSIS_METHODS
            ]
            print_metrics_table(f"{uni.upper()} — {scenario_label}", rows)

    for scenario, scenario_label in SCENARIOS:
        rows = [(MODEL_DISPLAY[m], results["per_model"][m]["pooled"][scenario])
                for m in ANALYSIS_METHODS]
        print_metrics_table(f"POOLED (ALL UNIVERSITIES) — {scenario_label}", rows)

    print_significance_block(results, "significance", "(bias shift − baseline)")
    print_significance_block(results, "significance_affine", "(affine − baseline)")
    print_significance_block(results, "significance_affine_vs_shift", "(affine − bias shift)")


# Main

def main():
    parser = argparse.ArgumentParser(description="Post-hoc bias corrections.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "output" / "bias_correction_results.json",
                        help="Path for the results JSON")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Paired-bootstrap iterations for ΔMAE/ΔQWK CIs (default 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conn = get_connection(args.db_path)
    cal_ids, splits_info = get_all_calibration_ids(conn)
    cal_data = load_complete_case_data(conn, include_ids=cal_ids)
    test_data = load_complete_case_data(conn, exclude_ids=cal_ids)
    conn.close()

    ticks_map = load_institution_ticks()
    universities = list(INSTITUTIONS.values())

    output: Dict = {
        "splits": {INSTITUTIONS[code]: info for code, info in splits_info.items()},
        "method": "bias_shift_and_affine",
        "bootstrap_iterations": args.bootstrap,
        "per_model": {},
    }

    for model in ANALYSIS_METHODS:
        cal_preds = np.concatenate([cal_data[u][model] for u in universities if u in cal_data])
        cal_human = np.concatenate([cal_data[u]["human"] for u in universities if u in cal_data])
        bias = fit_bias(cal_preds, cal_human)

        affine_params = {
            u: fit_affine(cal_data[u][model], cal_data[u]["human"])
            for u in universities if u in cal_data
        }

        test_human_per_uni = {u: test_data[u]["human"] for u in universities if u in test_data}
        test_preds_per_uni = {u: test_data[u][model] for u in universities if u in test_data}
        corrected_per_uni = {
            u: snap_to_ticks(test_preds_per_uni[u] + bias, ticks_map[u])
            for u in test_preds_per_uni
        }
        affine_per_uni = {
            u: snap_to_ticks(
                affine_params[u][0] * test_preds_per_uni[u] + affine_params[u][1],
                ticks_map[u],
            )
            for u in test_preds_per_uni
        }

        per_uni = {}
        for u in test_preds_per_uni:
            human = test_human_per_uni[u]
            per_uni[u] = {
                "baseline": compute_accuracy_metrics(human, test_preds_per_uni[u], ticks_map[u]),
                "corrected": compute_accuracy_metrics(human, corrected_per_uni[u], ticks_map[u]),
                "affine": compute_accuracy_metrics(human, affine_per_uni[u], ticks_map[u]),
            }

        pooled = {
            "baseline": compute_pooled_metrics(test_human_per_uni, test_preds_per_uni, ticks_map),
            "corrected": compute_pooled_metrics(test_human_per_uni, corrected_per_uni, ticks_map),
            "affine": compute_pooled_metrics(test_human_per_uni, affine_per_uni, ticks_map),
        }

        flat_human = np.concatenate([test_human_per_uni[u] for u in test_human_per_uni])
        flat_base = np.concatenate([test_preds_per_uni[u] for u in test_preds_per_uni])
        flat_corr = np.concatenate([corrected_per_uni[u] for u in corrected_per_uni])
        flat_affine = np.concatenate([affine_per_uni[u] for u in affine_per_uni])

        output["per_model"][model] = {
            "n_cal": int(len(cal_preds)),
            "bias": bias,
            "affine_params": {
                u: {"a": a, "b": b} for u, (a, b) in affine_params.items()
            },
            "per_university": per_uni,
            "pooled": pooled,
            "significance": paired_bootstrap_delta(
                flat_human, flat_base, flat_corr, n_iter=args.bootstrap, seed=args.seed),
            "significance_affine": paired_bootstrap_delta(
                flat_human, flat_base, flat_affine, n_iter=args.bootstrap, seed=args.seed),
            "significance_affine_vs_shift": paired_bootstrap_delta(
                flat_human, flat_corr, flat_affine, n_iter=args.bootstrap, seed=args.seed),
        }

    print_report(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(round_floats(output, PRECISION), f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
