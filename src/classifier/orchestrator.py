"""Orchestrator — coordinates model execution and result aggregation."""

import asyncio
from typing import List, Dict, Any, Optional, Union

from .logger import Logger
from .models.base_model import BaseModel
from .aggregators.base_aggregator import BaseAggregator


class Orchestrator:
    """Runs multiple LLM models on queries and aggregates their predictions.

    Args:
        models: List of model instances to query.
        aggregators: List of aggregator instances to combine predictions.
        logger: Logger instance.
        prompt: A single prompt string applied to all models, or a dict
            mapping model name → prompt for per-model prompts.
        task_config: Task configuration dict (for reference by callers).
    """

    def __init__(
        self,
        models: List[BaseModel],
        aggregators: List[BaseAggregator],
        logger: Optional[Logger] = None,
        prompt: Optional[Union[str, Dict[str, str]]] = None,
        task_config: Optional[dict] = None,
    ):
        self.models = models
        self.aggregators = aggregators
        self.logger = logger or Logger()
        self.task_config = task_config or {}

        # Support per-model prompts or a single shared prompt
        if isinstance(prompt, dict):
            self.prompts = prompt
        else:
            self.prompts = {model.name: (prompt or "") for model in models}

    def get_aggregators(self) -> List[BaseAggregator]:
        return self.aggregators

    def get_models(self) -> List[BaseModel]:
        return self.models

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_single(self, query: str) -> List[Dict[str, Any]]:
        """Send *query* to every model in parallel and return raw results.

        Returns:
            List of dicts, one per model, each containing at minimum
            ``classification`` and ``model`` keys.
        """
        self.logger.debug(f"Executing query: {query[:80]}...")

        tasks = [
            model.classify_content(
                {"type": "text", "content": query}, self.prompts[model.name]
            )
            for model in self.models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for model, result in zip(self.models, results):
            if isinstance(result, Exception):
                result = {"error": str(result)}
            result["model"] = model.name
            if "error" in result:
                self.logger.debug(f"Model {model.name} failed: {result['error']}")
            else:
                self.logger.debug(f"Model {model.name} completed successfully")
            processed.append(result)

        return processed

    async def execute(
        self, query: str, separate_results: bool = False
    ) -> Dict[str, Any]:
        """Execute a query and aggregate results across all aggregators.

        Args:
            query: The text to classify.
            separate_results: If True, also return raw model predictions.

        Returns:
            Dict keyed by aggregator name → aggregated result dict.
            If *separate_results* is True, additionally includes
            ``model_predictions`` key.
        """
        model_results = [await self.execute_single(query)]

        aggregated: Dict[str, Any] = {}
        for aggregator in self.aggregators:
            name = aggregator.get_name()
            self.logger.debug(f"Aggregating with: {name}")
            aggregated[name] = aggregator.aggregate(model_results)

        if separate_results:
            return {
                "model_predictions": model_results,
                "aggregated_results": aggregated,
            }
        return aggregated
