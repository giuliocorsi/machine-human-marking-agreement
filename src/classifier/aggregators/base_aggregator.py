"""Base class for all aggregation strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseAggregator(ABC):
    """Abstract base for aggregators that combine multiple model predictions."""

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def _filter_valid(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return only predictions without an ``error`` key."""
        return [p for p in predictions if not p.get("error")]

    @abstractmethod
    def aggregate(self, predictions: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Aggregate a batch of prediction sets.

        Args:
            predictions: Outer list = one per query. Inner list = one per model.

        Returns:
            One result dict per query with ``classification`` and ``confidence``.
        """
