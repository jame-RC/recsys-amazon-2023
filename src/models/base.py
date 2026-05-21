from typing import List


class BaseRecommender:
    def __init__(self, num_items: int):
        self.num_items = num_items
        self.name = "Base"

    def fit(self, train_data, monitor=None):
        raise NotImplementedError

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        raise NotImplementedError

    def score_all(self, history: List[int]):
        raise NotImplementedError
