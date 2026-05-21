"""Train BPR model with checkpointing and evaluation."""
import json
import os
import sys
import time
import torch

sys.path.insert(0, '.')

from train_monitor import TrainingMonitor
from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.models.bpr import BPRRecommender
from src.utils.config import TOP_K, RESULTS_DIR

CATEGORY = "Industrial_and_Scientific"

def main():
    print(f"Loading data for {CATEGORY}...", flush=True)
    dataset = AmazonDataset(CATEGORY)

    train_data = dataset.get_train_sequences()
    valid_data = dataset.get_eval_data("valid")
    test_data = dataset.get_eval_data("test")
    num_items = len(dataset.item_vocab)

    print(f"Items: {num_items}, Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}", flush=True)

    monitor = TrainingMonitor("bpr_metrics.json")
    monitor.on_start(category="Industrial_and_Scientific", model="BPR")

    model = BPRRecommender(num_items, num_epochs=50, lr=0.001)
    print(f"Device: {model.device}", flush=True)

    # Train
    model.fit(train_data, monitor=monitor)
    print("Training done.", flush=True)
    monitor.on_end()

    # Save model weights
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f"{CATEGORY}_bpr_model.pt")
    torch.save(model.model.state_dict(), model_path)
    print(f"Model saved to {model_path}", flush=True)

    # Evaluate on valid set
    print("Evaluating on validation set...", flush=True)
    t0 = time.time()
    valid_evaluator = Evaluator(model, valid_data, num_items, TOP_K)
    valid_metrics = valid_evaluator.evaluate()
    print(f"Valid done in {time.time()-t0:.1f}s", flush=True)

    # Evaluate on test set
    print("Evaluating on test set...", flush=True)
    t0 = time.time()
    test_evaluator = Evaluator(model, test_data, num_items, TOP_K)
    test_metrics = test_evaluator.evaluate()
    print(f"Test done in {time.time()-t0:.1f}s", flush=True)

    # Save results
    result = {
        "category": CATEGORY,
        "model": "BPR",
        "valid": valid_metrics,
        "test": test_metrics,
    }
    result_file = os.path.join(RESULTS_DIR, f"{CATEGORY}_bpr.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_file}", flush=True)

    # Print summary
    print(f"\n=== RESULTS ===")
    print(f"Valid NDCG@10: {valid_metrics[f'NDCG@{TOP_K}']:.4f}")
    print(f"Test  NDCG@10: {test_metrics[f'NDCG@{TOP_K}']:.4f}")


if __name__ == "__main__":
    main()
