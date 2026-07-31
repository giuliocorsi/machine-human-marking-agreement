#!/usr/bin/env python3

"""
Essay Scorer - Multi-University Support

Reads essays from opraise.db, scores them via LLM models,
and writes predictions back to the same DB rows.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import Orchestrator, Logger
from src.classifier.models import ClaudeModel, GptModel, GeminiModel
from src.classifier.aggregators import (
    MedianAggregator,
    SurprisinglyPopularAggregator,
    SimpleMeanAggregator,
    WeightedMeanAggregator,
)
from src.classifier.utils import load_yaml_config
from src.classifier.prompt_manager import build_model_prompt

from src.utils.scoring_utils import ScoringUtils
from src.utils.db_utils import get_connection, ensure_prediction_columns, save_predictions, DEFAULT_DB_PATH
from src.utils.calibration_utils import get_calibration_split
from src.prompts.prompt_renderer import PromptRenderer


class EssayScorer:
    """Reads essays from opraise.db, scores them, writes predictions back."""

    def __init__(self, university: str, task_file: str):
        self.university = university
        self.task_file = task_file
        self.task_config = None
        self.logger = None
        self.orchestrator = None

        self.prompt_renderer = PromptRenderer()

    def _load_task_config(self) -> None:
        try:
            self.task_config = load_yaml_config(self.task_file)
            print(f"Using task: {self.task_config['task']['name']}")
            print(f"Description: {self.task_config['task']['description']}")
        except Exception as e:
            print(f"Error loading task file '{self.task_file}': {e}")
            sys.exit(1)

    def _get_prompts(self, feedback: bool = False) -> Dict[str, str]:
        """Build per-model prompts using calibration best_prompts, or fall back to default."""
        best_prompts = self.task_config.get("calibration", {}).get("best_prompts", {})
        default_id = self.task_config.get("prompts", {}).get(
            "default_text_prompt_id", "A1_B3_C1"
        )

        model_prompts = {}
        for model_name in ["Gemini", "GPT", "Claude"]:
            prompt_id = best_prompts.get(model_name.lower(), default_id)
            model_prompts[model_name] = build_model_prompt(
                self.prompt_renderer, prompt_id, self.university, self.task_config,
                include_feedback=feedback,
            )
            print(f"  {model_name} prompt: {prompt_id}")

        return model_prompts

    def _initialise_components(self, feedback: bool = False) -> None:
        self.logger = Logger(log_level="INFO", console_output=True)

        models = [
            GeminiModel(os.getenv("GEMINI_MODEL"), name="Gemini"),
            GptModel(os.getenv("GPT_MODEL"), name="GPT"),
            ClaudeModel(os.getenv("CLAUDE_MODEL"), name="Claude"),
        ]

        calibration_rmse = self.task_config.get("calibration", {}).get("rmse", {})

        aggregators = [
            MedianAggregator(),
            SurprisinglyPopularAggregator(),
            SimpleMeanAggregator(),
        ]

        if calibration_rmse:
            aggregators.append(WeightedMeanAggregator(calibration_rmse))
        else:
            print("Warning: No calibration RMSE in task config — skipping weighted mean aggregator")

        prompts = self._get_prompts(feedback=feedback)

        self.orchestrator = Orchestrator(
            models, aggregators, self.logger, prompts, self.task_config,
        )

    async def score_essays(
        self,
        limit: Optional[int] = None,
        essay_ids: Optional[List[int]] = None,
        essay_type: Optional[str] = None,
        rescore: bool = False,
        feedback: bool = False,
        test_only: bool = False,
    ) -> None:
        """
        Score essays from opraise.db and write predictions back.

        Args:
            limit: Score at most N essays.
            essay_ids: Only score these specific DB row ids.
            essay_type: Filter by essay type ("long" or "short").
            rescore: If False (default), skip essays that already have predictions.
            test_only: If True, exclude calibration set essays (use for main evaluation).
        """
        self._load_task_config()
        self._initialise_components(feedback=feedback)

        conn = get_connection()
        ensure_prediction_columns(conn)

        # Build query — get the university name as stored in DB
        # (DB stores "Cambridge", CLI gets "cambridge")
        uni_row = conn.execute(
            "SELECT DISTINCT university FROM essays WHERE LOWER(university) = ?",
            (self.university.lower(),),
        ).fetchone()

        if uni_row is None:
            print(f"Error: No essays found for university '{self.university}' in opraise.db")
            conn.close()
            return

        uni_name = uni_row["university"]

        # Compute calibration exclusion set if --test-only
        exclude_ids: set = set()
        if test_only:
            cal_config = self.task_config.get("calibration", {})
            seed = cal_config.get("seed", 42)
            fraction = cal_config.get("fraction", 0.2)
            cal_rows, _ = get_calibration_split(conn, self.university, seed=seed, fraction=fraction)
            exclude_ids = {r["id"] for r in cal_rows}
            print(f"Test-only mode: excluding {len(exclude_ids)} calibration essays")
            # Calibration was performed on long essays only — restrict to match
            if not essay_type:
                essay_type = "long"
                print(f"  (filtering to long essays to match calibration split)")

        # Build the query
        query = "SELECT id, student_id, human_grade, human_grade_band, assignment_brief, assignment_content FROM essays WHERE university = ?"
        params: list = [uni_name]

        if essay_ids:
            placeholders = ",".join("?" * len(essay_ids))
            query += f" AND id IN ({placeholders})"
            params.extend(essay_ids)

        if essay_type:
            query += " AND LOWER(essay_type) = ?"
            params.append(essay_type.lower())

        if not rescore:
            # Skip rows that already have all model predictions filled
            query += " AND (gemini_score IS NULL OR gpt_score IS NULL OR claude_score IS NULL)"

        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            query += f" AND id NOT IN ({placeholders})"
            params.extend(sorted(exclude_ids))

        query += " ORDER BY id"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()

        if not rows:
            print("No essays to score (all already have predictions — use --rescore to overwrite).")
            conn.close()
            return

        total = len(rows)
        print(f"\nScoring {total} essays for {uni_name}...\n")

        semaphore = asyncio.Semaphore(20)
        scored = 0

        async def score_one(row):
            nonlocal scored
            async with semaphore:
                db_id = row["id"]
                text = row["assignment_content"]
                brief = row["assignment_brief"]
                if brief:
                    text = f"=== ASSIGNMENT BRIEF ===\n{brief}\n\n=== STUDENT SUBMISSION ===\n{text}"
                human_grade = row["human_grade"]
                human_band = row["human_grade_band"]

                result = await self.orchestrator.execute(text, separate_results=True)

                # Process model predictions
                model_preds = {}
                for data in result.get("model_predictions", [[]])[0]:
                    if data is None:
                        continue
                    model_name = data.get("model", "unknown")
                    if "error" in data:
                        self.logger.warning(f"Essay {db_id} - {model_name}: {data['error'][:80]}")
                        continue
                    classification = ScoringUtils.extract_grade(data.get("classification"))
                    band = ScoringUtils.get_grade_band(classification) if classification else None
                    model_preds[model_name] = {
                        "classification": classification,
                        "confidence": data.get("confidence"),
                        "grade_band": band,
                        "predicted_distribution": data.get("predicted_distribution", {}),
                        "feedback": data.get("feedback"),
                    }

                # Process aggregated results
                agg_results = {}
                for aggr in self.orchestrator.get_aggregators():
                    name = aggr.get_name()
                    aggr_result = result.get("aggregated_results", {}).get(name, {})
                    if isinstance(aggr_result, list) and aggr_result:
                        r = aggr_result[0]
                        if "error" not in r:
                            classification = ScoringUtils.extract_grade(r.get("classification"))
                            band = ScoringUtils.get_grade_band(classification) if classification else None
                            agg_results[name] = {
                                "classification": classification,
                                "confidence": r.get("confidence"),
                                "grade_band": band,
                            }

                save_predictions(conn, db_id, model_preds, agg_results)
                scored += 1

                # Brief summary
                median = agg_results.get("median_vote", {})
                median_score = median.get("classification", "?")
                median_band = median.get("grade_band", "?")
                print(f"  [{scored}/{total}] Essay {db_id} (student {row['student_id']}): "
                      f"human={human_grade}/{human_band} → median={median_score}/{median_band}")

        tasks = [score_one(row) for row in rows]
        await asyncio.gather(*tasks)

        conn.close()
        print(f"\nDone. Saved predictions for {scored}/{total} essays to opraise.db")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Score essays from opraise.db.")
    parser.add_argument("university", help="University name (e.g., cambridge, mmu, nottingham)")
    parser.add_argument("--type", "-t", choices=["long", "short"], default=None, help="Filter by essay type")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Score at most N essays")
    parser.add_argument("--ids", nargs="+", type=int, default=None, help="Score specific essay DB row ids")
    parser.add_argument("--rescore", action="store_true", help="Re-score essays that already have predictions")
    parser.add_argument("--feedback", action="store_true", help="Include written feedback from each model")
    parser.add_argument("--test-only", action="store_true", help="Exclude calibration set essays (test set only)")
    args = parser.parse_args()

    university = args.university.lower()

    # Validate university is supported by PromptRenderer
    renderer = PromptRenderer()
    if university not in renderer.list_institutions():
        print(f"Error: University '{university}' not supported.")
        print(f"Available: {renderer.list_institutions()}")
        sys.exit(1)

    # Find task config
    task_file = PROJECT_ROOT / "src" / "config" / "institutions" / f"{university}.yaml"
    if not task_file.exists():
        print(f"Error: Task config not found: {task_file}")
        sys.exit(1)

    if not DEFAULT_DB_PATH.exists():
        print(f"Error: Database not found at {DEFAULT_DB_PATH}")
        sys.exit(1)

    print(f"{'=' * 70}")
    print(f"ESSAY SCORER")
    print(f"University: {university}")
    print(f"Database:   {DEFAULT_DB_PATH}")
    if args.type:
        print(f"Type:       {args.type}")
    if args.limit:
        print(f"Limit:      {args.limit}")
    if args.ids:
        print(f"IDs:        {args.ids}")
    if args.rescore:
        print(f"Rescore:    yes")
    if args.feedback:
        print(f"Feedback:   yes")
    if args.test_only:
        print(f"Test-only:  yes (calibration essays excluded)")
    print(f"{'=' * 70}")

    scorer = EssayScorer(university, str(task_file))
    await scorer.score_essays(
        limit=args.limit, essay_ids=args.ids, essay_type=args.type,
        rescore=args.rescore, feedback=args.feedback, test_only=args.test_only,
    )


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
