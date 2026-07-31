"""Weighted mean aggregator — weights each model by inverse calibration RMSE."""

import statistics
from typing import List, Dict, Any

from .base_aggregator import BaseAggregator


class WeightedMeanAggregator(BaseAggregator):
    """Aggregates by taking the weighted mean of numeric predictions.

    Weights follow Cooke's Classical Model: each model's weight is proportional
    to the inverse of its RMSE on the calibration set.

        wₖ = (1 / RMSEₖ) / Σⱼ (1 / RMSEⱼ)

    The calibration RMSE dict maps model name prefixes (lowercase) to RMSE values.
    """

    def __init__(self, calibration_rmse: Dict[str, float]):
        """Initialise with per-model calibration RMSE values.

        Args:
            calibration_rmse: Mapping of lowercase model name prefix to RMSE.
                e.g. {"gemini": 8.2, "gpt": 7.5, "claude": 6.9}
        """
        super().__init__("weighted_mean")
        if not calibration_rmse:
            raise ValueError("calibration_rmse must be a non-empty dict")
        if any(v <= 0 for v in calibration_rmse.values()):
            raise ValueError("All RMSE values must be positive")

        inverse_sum = sum(1.0 / rmse for rmse in calibration_rmse.values())
        self.weights = {
            name.lower(): (1.0 / rmse) / inverse_sum
            for name, rmse in calibration_rmse.items()
        }

    def aggregate(self, predictions: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        try:
            results = []
            for prediction_set in predictions:
                valid = self._filter_valid(prediction_set)
                if not valid:
                    results.append({"error": "All models failed"})
                    continue

                weighted_sum = 0.0
                total_weight = 0.0
                values = []

                for p in valid:
                    try:
                        value = float(p["classification"])
                    except (ValueError, TypeError, KeyError):
                        continue

                    model_key = self._resolve_model_key(p.get("model", ""))
                    weight = self.weights.get(model_key)
                    if weight is None:
                        continue

                    weighted_sum += weight * value
                    total_weight += weight
                    values.append(value)

                if total_weight == 0:
                    results.append({"error": "No matching weighted classifications"})
                    continue

                # Renormalise in case some models failed
                weighted_val = weighted_sum / total_weight

                # Confidence: inverse of weighted stdev (higher agreement = higher confidence)
                if len(values) > 1:
                    stdev = statistics.stdev(values)
                    confidence = round(max(0, 100 - stdev))
                else:
                    confidence = 100

                results.append({
                    "classification": str(round(weighted_val)),
                    "confidence": confidence,
                })
            return results
        except Exception as e:
            return [{"error": str(e)}]
    @staticmethod
    def _resolve_model_key(model_name: str) -> str:
        """Map a model display name to its weight-dict key."""
        name = model_name.lower()
        for prefix in ("gemini", "gpt", "claude"):
            if prefix in name:
                return prefix
        return name
