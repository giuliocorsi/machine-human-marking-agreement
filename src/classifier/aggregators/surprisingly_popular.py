"""Surprisingly Popular aggregator.

Averages the ``predicted_distribution`` across models, normalises,
and picks the class with the highest aggregated probability.
"""

from typing import List, Dict, Any

from .base_aggregator import BaseAggregator


class SurprisinglyPopularAggregator(BaseAggregator):
    """Uses averaged predicted distributions to determine the final class."""

    def __init__(self):
        super().__init__("surprisingly_popular")

    def aggregate(self, predictions: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        try:
            results = []
            for prediction_set in predictions:
                valid = [
                    p for p in prediction_set
                    if "predicted_distribution" in p and not p.get("error")
                ]
                if not valid:
                    results.append({"error": "No distribution data"})
                    continue

                result = self._aggregate_distributions(valid)
                results.append(result)
            return results
        except Exception as e:
            return [{"error": str(e)}]

    @staticmethod
    def _aggregate_distributions(valid: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average distributions across models, normalise, and classify."""
        # Sum totals across all models
        totals: Dict[str, float] = {}
        for p in valid:
            for key, value in p["predicted_distribution"].items():
                try:
                    totals[key] = totals.get(key, 0.0) + float(value)
                except (ValueError, TypeError):
                    continue

        if not totals:
            return {"error": "No totals to normalise"}

        total_sum = sum(totals.values())
        if total_sum == 0:
            return {"error": "All totals are zero"}

        # Normalise to percentages
        normalised = {
            key: round((value / total_sum) * 100, 1)
            for key, value in totals.items()
        }

        best_class = max(normalised, key=normalised.get)
        return {
            "classification": best_class,
            "confidence": normalised[best_class],
            "distributions": normalised,
        }
