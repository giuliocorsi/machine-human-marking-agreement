"""Simple mean aggregator — takes the arithmetic mean of numeric classifications."""

import statistics
from typing import List, Dict, Any

from .base_aggregator import BaseAggregator


class SimpleMeanAggregator(BaseAggregator):
    """Aggregates by taking the arithmetic mean of numeric predictions."""

    def __init__(self):
        super().__init__("simple_mean")

    def aggregate(self, predictions: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        try:
            results = []
            for prediction_set in predictions:
                valid = self._filter_valid(prediction_set)
                if not valid:
                    results.append({"error": "All models failed"})
                    continue

                values = []
                for p in valid:
                    try:
                        values.append(float(p["classification"]))
                    except (ValueError, TypeError, KeyError):
                        continue

                if not values:
                    results.append({"error": "No numeric classifications"})
                    continue

                mean_val = statistics.mean(values)
                # Confidence: inverse of coefficient of variation (higher = more agreement)
                if len(values) > 1:
                    stdev = statistics.stdev(values)
                    confidence = round(max(0, 100 - stdev)) if mean_val else 0
                else:
                    confidence = 100

                results.append({
                    "classification": str(round(mean_val)),
                    "confidence": confidence,
                })
            return results
        except Exception as e:
            return [{"error": str(e)}]
