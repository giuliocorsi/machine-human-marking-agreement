"""Median aggregator — takes the median of numeric classifications."""

import statistics
from typing import List, Dict, Any

from .base_aggregator import BaseAggregator


class MedianAggregator(BaseAggregator):
    """Aggregates by taking the statistical median of numeric predictions."""

    def __init__(self):
        super().__init__("median_vote")

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

                median_val = statistics.median(values)
                # Confidence: percentage of values within ±5 of the median
                close = sum(1 for v in values if abs(v - median_val) <= 5)
                confidence = round(100 * close / len(values))

                results.append({
                    "classification": str(int(median_val)) if median_val == int(median_val) else str(median_val),
                    "confidence": confidence,
                })
            return results
        except Exception as e:
            return [{"error": str(e)}]
