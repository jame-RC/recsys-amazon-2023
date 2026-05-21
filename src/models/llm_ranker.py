from typing import List, Optional

from src.llm.api import LLMClient
from src.llm.prompt import (
    FEW_SHOT_RANKING, ZERO_SHOT_RANKING, ZERO_SHOT_SYSTEM,
    format_candidates, format_history,
)
from src.models.base import BaseRecommender
from src.utils.config import LLM_CANDIDATE_SIZE, TOP_K
from src.utils.logger import get_logger

logger = get_logger("llm_ranker")


class LLMRankerRecommender(BaseRecommender):
    def __init__(self, num_items: int, base_model: BaseRecommender,
                 llm_client: LLMClient, candidate_size: int = LLM_CANDIDATE_SIZE,
                 use_few_shot: bool = False, item_meta: Optional[dict] = None):
        super().__init__(num_items)
        self.name = "LLM_Ranker" + ("_FewShot" if use_few_shot else "_ZeroShot")
        self.base_model = base_model
        self.llm_client = llm_client
        self.candidate_size = candidate_size
        self.use_few_shot = use_few_shot
        self.item_meta = item_meta or {}

    def _get_item_info(self, item_idx: int) -> dict:
        for asin, meta in self.item_meta.items():
            if meta.get("idx") == item_idx:
                return {"id": item_idx, "title": meta.get("title", f"Item_{item_idx}"),
                        "description": meta.get("description", "")}
        return {"id": item_idx, "title": f"Item_{item_idx}", "description": ""}

    def _build_history_text(self, history: List[int]) -> str:
        items = [self._get_item_info(idx) for idx in history[-10:]]
        return format_history(items)

    def _build_candidate_text(self, candidates: List[int]) -> str:
        items = [{"id": i + 1, "title": self._get_item_info(c)["title"]} for i, c in enumerate(candidates)]
        return format_candidates(items)

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        if not history:
            return []

        base_candidates = self.base_model.recommend(history, self.candidate_size)
        if not base_candidates:
            return []

        history_text = self._build_history_text(history)
        candidate_text = self._build_candidate_text(base_candidates)

        if self.use_few_shot:
            prompt = FEW_SHOT_RANKING.format(history=history_text, candidates=candidate_text)
        else:
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
