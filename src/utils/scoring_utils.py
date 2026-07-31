#!/usr/bin/env python3

"""
Shared scoring utilities for essay classification.
UK grade band conversion (First, Upper Second, Lower Second, Third, Fail).
Tick-based tolerance calculations for institutional grade scales.
"""

from typing import Optional, Union, List
import bisect


class ScoringUtils:
    """
    Handles UK grade band conversions and validations.
    Uses standard UK grading system with text names.
    """

    GRADE_BANDS = {
        'First': (70, 100),
        'Upper Second': (60, 69),
        'Lower Second': (50, 59),
        'Third': (40, 49),
        'Fail': (0, 39)
    }

    # Mapping of equivalent grade band names to canonical form
    BAND_ALIASES = {
        'first': 'First',
        '1st': 'First',
        '1': 'First',
        'upper second': 'Upper Second',
        'upper-second': 'Upper Second',
        'uppersecond': 'Upper Second',
        '2:1': 'Upper Second',
        '2.1': 'Upper Second',
        '21': 'Upper Second',
        'lower second': 'Lower Second',
        'lower-second': 'Lower Second',
        'lowersecond': 'Lower Second',
        '2:2': 'Lower Second',
        '2.2': 'Lower Second',
        '22': 'Lower Second',
        'third': 'Third',
        '3rd': 'Third',
        '3': 'Third',
        'fail': 'Fail',
        'f': 'Fail',
    }

    @staticmethod
    def extract_grade(grade_value) -> Optional[int]:
        """
        Extract integer grade from various formats.
        Handles '58/100', 58, 58.0, '58', etc.
        """
        if grade_value is None:
            return None
        try:
            if isinstance(grade_value, str) and "/" in grade_value:
                return int(grade_value.split("/")[0])
            return int(float(grade_value))
        except (ValueError, TypeError):
            return None

    @classmethod
    def normalize_band(cls, band: Optional[str]) -> Optional[str]:
        """
        Normalize a grade band name to canonical form.
        Handles variations like '2:1' -> 'Upper Second', '2:2' -> 'Lower Second'.
        """
        if band is None:
            return None
        band_lower = band.lower().strip()
        return cls.BAND_ALIASES.get(band_lower, band)

    @classmethod
    def get_grade_band(cls, score: Optional[Union[int, float, str]]) -> Optional[str]:
        """
        Convert numerical score to UK grade band.
        """
        if score is None:
            return None

        try:
            score_value = float(score)

            # Validate score range
            if score_value < 0 or score_value > 100:
                return None

            # Determine grade band
            if score_value >= 70:
                return "First"
            elif score_value >= 60:
                return "Upper Second"
            elif score_value >= 50:
                return "Lower Second"
            elif score_value >= 40:
                return "Third"
            else:
                return "Fail"

        except (ValueError, TypeError):
            return None

    @classmethod
    def validate_score(cls, score: Union[int, float]) -> bool:
        """
        Validate that a score is within acceptable range.
        """
        try:
            score_value = float(score)
            return 0 <= score_value <= 100
        except (ValueError, TypeError):
            return False

    @classmethod
    def get_band_range(cls, band_name: str) -> Optional[tuple]:
        """
        Get the score range for a given grade band.
        """
        return cls.GRADE_BANDS.get(band_name)

    @staticmethod
    def find_nearest_tick(score: float, ticks: List[int]) -> int:
        """
        Find the nearest tick to a given score.
        Useful for median/mean values that may fall between valid ticks.
        """
        if not ticks:
            return int(round(score))

        # Binary search for insertion point
        pos = bisect.bisect_left(ticks, score)

        if pos == 0:
            return ticks[0]
        if pos == len(ticks):
            return ticks[-1]

        # Compare distance to adjacent ticks
        before = ticks[pos - 1]
        after = ticks[pos]

        if score - before <= after - score:
            return before
        return after

    @staticmethod
    def get_tick_index(score: float, ticks: List[int]) -> int:
        """
        Get the index of the nearest tick for a given score.
        """
        nearest = ScoringUtils.find_nearest_tick(score, ticks)
        return ticks.index(nearest)

    @staticmethod
    def tick_distance(score1: float, score2: float, ticks: List[int]) -> int:
        """
        Calculate the distance in tick positions between two scores.
        Both scores are first mapped to their nearest ticks.
        """
        idx1 = ScoringUtils.get_tick_index(score1, ticks)
        idx2 = ScoringUtils.get_tick_index(score2, ticks)
        return abs(idx1 - idx2)

    @staticmethod
    def scores_within_tick_tolerance(score1: float, score2: float, ticks: List[int], tolerance: int) -> bool:
        """
        Check if two scores are within a given tick tolerance of each other.
        """
        return ScoringUtils.tick_distance(score1, score2, ticks) <= tolerance

    @staticmethod
    def get_adjacent_ticks(score: float, ticks: List[int], tolerance: int) -> List[int]:
        """
        Get all ticks within ±tolerance tick positions of a score.
        """
        idx = ScoringUtils.get_tick_index(score, ticks)
        start = max(0, idx - tolerance)
        end = min(len(ticks), idx + tolerance + 1)
        return ticks[start:end]
