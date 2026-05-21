from typing import List, Optional

from src.llm.api import LLMClient
from src.llm.prompt import ZERO_SHOT_RANKING, ZERO_SHOT_SYSTEM, format_candidates, format_history
from src.models.base import BaseRecommender
from src.utils.config import LLM_CANDIDATE_SIZE, TOP_K
from src.utils.logger import get_logger

logger = get_logger("llm_hybrid")


class LLMHybridRecommender(BaseRecommender):
    def __init__(self, num_items: int, models: List[BaseRecommender],
                 llm_client: LLMClient, weights: Optional[List[float]] = None,
                 candidate_size: int = LLM_CANDIDATE_SIZE,
                 item_meta: Optional[dict] = None):
        super().__init__(num_items)
        self.name = "LLM_Hybrid"
        self.models = models
        self.llm_client = llm_client
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.candidate_size = candidate_size
        self.item_meta = item_meta or {}

    def _get_item_info(self, item_idx: int) -> dict:
        for asin, meta in self.item_meta.items():
            if meta.get("idx") == item_idx:
                return {"id": item_idx, "title": meta.get("title", f"Item_{item_idx}"),
                        "description": meta.get("description", "")}
        return {"id": item_idx, "title": f"Item_{item_idx}", "description": ""}

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        if not history:
            return []

        candidate_scores = {}
        for model, weight in zip(self.models, self.weights):
            candidates = model.recommend(history, self.candidate_size)
            for rank, item in enumerate(candidates):
                score = weight * (1.0 / (rank + 1))
                candidate_scores[item] = candidate_scores.get(item, 0) + score

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        base_candidates = [item for item, _ in sorted_candidates[:self.candidate_size]]

        if not base_candidates:
            return []

        history_items = [self._get_item_info(idx) for idx in history[-10:]]
        history_text = format_history(history_items)
        candidate_items = [{"id": i + 1, "title": self._get_item_info(c)["title"]} for i, c in enumerate(base_candidates)]
        candidate_text = format_candidates(candidate_items)

        prompt = ZERO_SHOT_RANKING.format(history=history_text, candidates=candidate_text)
        ranked_indices = self.llm_client.rank(ZERO_SHOT_SYSTEM, prompt, len(base_candidates))

        result = []
        for idx in ranked_indices:
            if 1 <= idx <= len(base_candidates):
                item = base_candidates[idx - 1]
                if item not in result:
                    result.append(item)
            if len(result) >= top_k:
                break
        return result
