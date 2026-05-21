from typing import List
from collections import Counter

from src.models.base import BaseRecommender


class PopRecommender(BaseRecommender):
    def __init__(self, num_items: int):
        super().__init__(num_items)
        self.name = "Popularity"
        self.item_counts = None
        self.popular_items = None

    def fit(self, train_data, **kwargs):
        counts = Counter()
        for _, history, target in train_data:
            counts[target] += 1
            for item in history:
                counts[item] += 1
        self.item_counts = counts
        self.popular_items = [item for item, _ in counts.most_common()]

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        history_set = set(history)
        result = []
        for item in self.popular_items:
            if item not in history_set:
                result.append(item)
            if len(result) >= top_k:
                break
        return result
