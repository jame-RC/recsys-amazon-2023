import math
from collections import defaultdict
from typing import List

import numpy as np

from src.models.base import BaseRecommender


class ItemCFRecommender(BaseRecommender):
    def __init__(self, num_items: int, top_n: int = 50):
        super().__init__(num_items)
        self.name = "ItemCF"
        self.top_n = top_n
        self.item_sim = None

    def fit(self, train_data, **kwargs):
        co_occur = defaultdict(lambda: defaultdict(int))
        item_freq = defaultdict(int)

        for _, history, target in train_data:
            items = history + [target]
            for i, item_i in enumerate(items):
                item_freq[item_i] += 1
                for j in range(i + 1, len(items)):
                    item_j = items[j]
                    co_occur[item_i][item_j] += 1
                    co_occur[item_j][item_i] += 1

        self.item_sim = {}
        for item_i, neighbors in co_occur.items():
            sims = []
            for item_j, count in neighbors.items():
                denom = math.sqrt(item_freq[item_i] * item_freq[item_j])
                if denom > 0:
                    sims.append((item_j, count / denom))
            sims.sort(key=lambda x: -x[1])
            self.item_sim[item_i] = sims[:self.top_n]

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        if not history:
            return []

        scores = defaultdict(float)
        history_set = set(history)

        for item in history:
            if item in self.item_sim:
                for neighbor, sim in self.item_sim[item]:
                    if neighbor not in history_set:
                        scores[neighbor] += sim

        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return [item for item, _ in sorted_items[:top_k]]
