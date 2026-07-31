"""Input handler — processes essay entries through the orchestrator."""

import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional

from .logger import Logger
from .utils import save_data


@dataclass
class ModelPrediction:
    classification: Optional[str] = None
    confidence: Optional[float] = None
    predicted_distribution: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AggregatedResult:
    classification: Optional[str] = None
    confidence: Optional[float] = None
    distributions: Optional[Any] = None
    votes: Optional[Dict[str, int]] = None
    error: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


class InputHandler:
    """Processes structured input data through an Orchestrator.

    Iterates over ``entries`` in the input data, sends each entry's text
    to the orchestrator, and assembles the results.
    """

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle(
        self,
        orchestrator,
        queries,            # kept for API compat (unused — pass None)
        input_data: Dict[str, Any],
        path_dir: str,
        output_file: str,
    ) -> Dict[str, Any]:
        """Process all entries and return the full results dict."""
        if input_data is None:
            raise ValueError("No input data provided.")

        results = self._prepare_results_container(input_data)

        semaphore = asyncio.Semaphore(20)

        async def _process(entry):
            async with semaphore:
                return await self._process_single_entry(entry, orchestrator)

        tasks = [_process(entry) for entry in input_data["entries"]]
        results["entries"] = await asyncio.gather(*tasks)

        # Save intermediate results
        save_data(output_file, results)
        self.logger.info(f"Results saved to {output_file}")

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_results_container(input_data: Dict[str, Any]) -> Dict[str, Any]:
        info = input_data["dataset_info"].copy()
        info["name"] = "Results " + info["name"]
        info["version"] = str(round(float(info.get("version", "1.0")) + 0.1, 1))
        info["created_date"] = datetime.now().strftime("%Y-%m-%d")
        return {"dataset_info": info}

    async def _process_single_entry(self, entry: Dict, orchestrator) -> Dict:
        self.logger.info(f"Processing entry ID: {entry.get('id', '?')}")

        query = entry["text"]
        result = await orchestrator.execute(query, separate_results=True)

        # Process model predictions
        models_results = {}
        for data in result.get("model_predictions", [[]])[0]:
            if data is None:
                continue
            model_name = data.get("model", "unknown")
            if "error" in data:
                models_results[model_name] = ModelPrediction(error=data["error"])
                self.logger.warning(f"Model {model_name} error: {data['error']}")
            else:
                models_results[model_name] = ModelPrediction(
                    classification=data.get("classification"),
                    confidence=data.get("confidence"),
                    predicted_distribution=data.get("predicted_distribution"),
                )

        # Process aggregated results
        aggregate_results = {}
        for aggr in orchestrator.get_aggregators():
            name = aggr.get_name()
            aggr_result = result.get("aggregated_results", {}).get(name, {})
            if isinstance(aggr_result, list) and aggr_result:
                r = aggr_result[0]
                if "error" in r:
                    aggregate_results[name] = AggregatedResult(error=r["error"])
                else:
                    aggregate_results[name] = AggregatedResult(
                        classification=r.get("classification"),
                        confidence=r.get("confidence"),
                        distributions=r.get("distributions"),
                        votes=r.get("votes"),
                    )
            else:
                aggregate_results[name] = AggregatedResult(error="No results available")

        processed = entry.copy()
        processed.update({
            "model_predictions": {k: v.to_dict() for k, v in models_results.items()},
            "aggregated_results": {k: v.to_dict() for k, v in aggregate_results.items()},
            "status": "error" if any(
                "error" in v.to_dict() for v in models_results.values()
            ) else "success",
        })
        return processed
