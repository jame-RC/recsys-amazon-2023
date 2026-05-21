from typing import List, Optional

import numpy as np

from src.llm.api import LLMClient
from src.llm.prompt import RAG_PROMPT, RAG_SYSTEM, format_candidates, format_context, format_history
from src.models.base import BaseRecommender
from src.utils.config import LLM_CANDIDATE_SIZE, TOP_K
from src.utils.logger import get_logger

logger = get_logger("llm_rag")


class LLMRAGRecommender(BaseRecommender):
    def __init__(self, num_items: int, base_model: BaseRecommender,
                 llm_client: LLMClient, item_meta: dict,
                 candidate_size: int = LLM_CANDIDATE_SIZE):
        super().__init__(num_items)
        self.name = "LLM_RAG"
        self.base_model = base_model
        self.llm_client = llm_client
        self.item_meta = item_meta
        self.candidate_size = candidate_size

        self.item_embeddings = None
        self.item_ids = None
        self._build_index()

    def _build_index(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")

            texts = []
            self.item_ids = []
            for asin, meta in self.item_meta.items():
                title = meta.get("title", "")
                desc = meta.get("description", "")[:200]
                text = f"{title} {desc}".strip()
                if text and meta.get("idx"):
                    texts.append(text)
                    self.item_ids.append(meta["idx"])

            if texts:
                self.item_embeddings = model.encode(texts, show_progress_bar=False)
                logger.info(f"Built RAG index with {len(texts)} items")
        except Exception as e:
            logger.warning(f"Failed to build RAG index: {e}")

    def _retrieve(self, history: List[int], top_k: int = 5) -> List[dict]:
        if self.item_embeddings is None or not history:
            return []

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")

            history_texts = []
            for idx in history[-5:]:
                for asin, meta in self.item_meta.items():
                    if meta.get("idx") == idx:
                        title = meta.get("title", "")
                        history_texts.append(title)
                        break

            if not history_texts:
                return []

            query = " ".join(history_texts)
            query_emb = model.encode([query], show_progress_bar=False)

            scores = np.dot(self.item_embeddings, query_emb.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                item_idx = self.item_ids[idx]
                for asin, meta in self.item_meta.items():
                    if meta.get("idx") == item_idx:
                        results.append({
                            "title": meta.get("title", ""),
                            "description": meta.get("description", "")[:200],
                            "categories": meta.get("categories", []),
                        })
                        break
            return results
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return []

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        if not history:
            return []

        base_candidates = self.base_model.recommend(history, self.candidate_size)
        if not base_candidates:
            return []

        retrieved = self._retrieve(history)
        context_text = format_context(retrieved)

        history_items = []
        for idx in history[-10:]:
            for asin, meta in self.item_meta.items():
                if meta.get("idx") == idx:
                    history_items.append({"title": meta.get("title", ""), "description": meta.get("description", "")})
                    break
        history_text = format_history(history_items)

        candidate_items = [{"id": i + 1, "title": self._get_title(c)} for i, c in enumerate(base_candidates)]
        candidate_text = format_candidates(candidate_items)

        prompt = RAG_PROMPT.format(history=history_text, context=context_text, candidates=candidate_text)
        ranked_indices = self.llm_client.rank(RAG_SYSTEM, prompt, len(base_candidates))

        result = []
        for idx in ranked_indices:
            if 1 <= idx <= len(base_candidates):
                item = base_candidates[idx - 1]
                if item not in result:
                    result.append(item)
            if len(result) >= top_k:
                break
        return result

    def _get_title(self, item_idx: int) -> str:
        for asin, meta in self.item_meta.items():
            if meta.get("idx") == item_idx:
                return meta.get("title", f"Item_{item_idx}")
        return f"Item_{item_idx}"
