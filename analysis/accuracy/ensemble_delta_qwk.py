#!/usr/bin/env python3

"""
Ensemble vs Best Individual Model

Paired bootstrap over essays: ΔQWK = QWK(wmean) − QWK(best individual model)
per institution, with percentile 95% CIs. CIs spanning zero mean the ensemble
is statistically indistinguishable from the best model.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db_utils import get_connection
from src.utils.calibration_utils import get_all_calibration_ids
from src.utils.analysis_utils import INSTITUTIONS, load_complete_case_data


MODELS = ("claude", "gemini", "gpt")
ENSEMBLE = "wmean"


def qwk(human: np.ndarray, preds: np.ndarray) -> float:
    """Quadratic-weighted Cohen's kappa on integer marks."""
    return cohen_kappa_score(human.astype(int), preds.astype(int), weights="quadratic")


def paired_bootstrap_delta(
    human: np.ndarray, ensemble: np.ndarray, comparator: np.ndarray,
    n_iter: int, seed: int,
) -> Dict:
    """Percentile 95% CI on ΔQWK = QWK(ensemble) − QWK(comparator), resampling essays."""
    rng = np.random.default_rng(seed)
    n = len(human)
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        try:
            deltas[i] = qwk(human[idx], ensemble[idx]) - qwk(human[idx], comparator[idx])
        except ValueError:
            deltas[i] = np.nan
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_qwk": float(qwk(human, ensemble) - qwk(human, comparator)),
        "ci95": [float(lo), float(hi)],
        "spans_zero": bool(lo <= 0.0 <= hi),
        "n_boot": int(deltas.size),
    }


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap ΔQWK: ensemble vs best model.")
    parser.add_argument("--bootstrap", type=int, default=10_000,
                        help="Bootstrap resamples (default 10000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "output" / "ensemble_delta_qwk.json",
                        help="Path for the results JSON")
    args = parser.parse_args()

    conn = get_connection()
    exclude_ids, _ = get_all_calibration_ids(conn)
    data = load_complete_case_data(conn, exclude_ids=exclude_ids)
    conn.close()

    results: Dict = {"n_bootstrap": args.bootstrap, "seed": args.seed, "institutions": {}}

    print(f"Paired bootstrap ΔQWK (wmean − comparator), {args.bootstrap} resamples, "
          f"seed {args.seed}\n")
    for uni in INSTITUTIONS.values():
        block = data[uni]
        human = block["human"]
        model_qwks = {m: qwk(human, block[m]) for m in MODELS}
        best = max(model_qwks, key=model_qwks.get)

        entry = {
            "n": len(human),
            "qwk": {ENSEMBLE: qwk(human, block[ENSEMBLE]), **model_qwks},
            "best_model": best,
            "vs_best": paired_bootstrap_delta(
                human, block[ENSEMBLE], block[best], args.bootstrap, args.seed),
            "vs_each": {m: paired_bootstrap_delta(
                human, block[ENSEMBLE], block[m], args.bootstrap, args.seed) for m in MODELS},
        }
        results["institutions"][uni] = entry

        d = entry["vs_best"]
        print(f"{uni} (n = {entry['n']})")
        print(f"  QWK: " + "  ".join(f"{m} {q:.3f}" for m, q in entry["qwk"].items()))
        print(f"  best individual: {best} ({model_qwks[best]:.3f})")
        print(f"  ΔQWK (wmean − {best}) = {d['delta_qwk']:+.3f}  "
              f"95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]  "
              f"{'spans zero' if d['spans_zero'] else 'EXCLUDES ZERO'}")
        for m in MODELS:
            dm = entry["vs_each"][m]
            print(f"    vs {m:<7} Δ = {dm['delta_qwk']:+.3f}  "
                  f"CI [{dm['ci95'][0]:+.3f}, {dm['ci95'][1]:+.3f}]")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
