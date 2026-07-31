"""Aggregation strategies for combining model predictions."""

from .base_aggregator import BaseAggregator
from .median import MedianAggregator
from .simple_mean import SimpleMeanAggregator
from .surprisingly_popular import SurprisinglyPopularAggregator
from .weighted_mean import WeightedMeanAggregator

__all__ = [
    "BaseAggregator",
    "MedianAggregator",
    "SimpleMeanAggregator",
    "SurprisinglyPopularAggregator",
    "WeightedMeanAggregator",
]
