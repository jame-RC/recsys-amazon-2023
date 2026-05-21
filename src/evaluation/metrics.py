import math
from typing import List


def ndcg_at_k(ranked_list: List[int], target: int, k: int = 10) -> float:
    for i, item in enumerate(ranked_list[:k]):
        if item == target:
            return 1.0 / math.log2(i + 2)
    return 0.0


def hit_at_k(ranked_list: List[int], target: int, k: int = 10) -> float:
    return 1.0 if target in ranked_list[:k] else 0.0


def recall_at_k(ranked_list: List[int], target: int, k: int = 10) -> float:
    return hit_at_k(ranked_list, target, k)


def mrr(ranked_list: List[int], target: int, k: int = 10) -> float:
    for i, item in enumerate(ranked_list[:k]):
        if item == target:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_batch(predictions: List[List[int]], targets: List[int], k: int = 10) -> dict:
    ndcg_scores = []
    hit_scores = []
    mrr_scores = []

    for pred, target in zip(predictions, targets):
        ndcg_scores.append(ndcg_at_k(pred, target, k))
        hit_scores.append(hit_at_k(pred, target, k))
        mrr_scores.append(mrr(pred, target, k))

    return {
        f"NDCG@{k}": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        f"Hit@{k}": sum(hit_scores) / len(hit_scores) if hit_scores else 0.0,
        f"MRR@{k}": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
    }
