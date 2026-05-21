import time
from typing import List, Tuple

from src.evaluation.metrics import evaluate_batch
from src.utils.config import TOP_K
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class Evaluator:
    def __init__(self, model, test_data: List[Tuple[int, List[int], int]], num_items: int, top_k: int = TOP_K):
        self.model = model
        self.test_data = test_data
        self.num_items = num_items
        self.top_k = top_k

    def evaluate(self, batch_size: int = 512) -> dict:
        predictions = []
        targets = []

        # Use recommend_batch if available, otherwise fall back to per-user recommend
        has_batch = hasattr(self.model, "recommend_batch")

        start_time = time.time()
        if has_batch:
            # Process in batches for memory efficiency
            for i in range(0, len(self.test_data), batch_size):
                batch = self.test_data[i:i + batch_size]
                histories = [h for _, h, _ in batch]
                batch_preds = self.model.recommend_batch(histories, self.top_k)
                predictions.extend(batch_preds)
                targets.extend([t for _, _, t in batch])
        else:
            for user_idx, history, target in self.test_data:
                pred = self.model.recommend(history, self.top_k)
                predictions.append(pred)
                targets.append(target)

        elapsed = time.time() - start_time
        metrics = evaluate_batch(predictions, targets, self.top_k)
        metrics["eval_time_sec"] = round(elapsed, 2)

        logger.info(
            f"NDCG@{self.top_k}={metrics[f'NDCG@{self.top_k}']:.4f} "
            f"Hit@{self.top_k}={metrics[f'Hit@{self.top_k}']:.4f} "
            f"MRR@{self.top_k}={metrics[f'MRR@{self.top_k}']:.4f} "
            f"Time={elapsed:.1f}s"
        )
        return metrics
