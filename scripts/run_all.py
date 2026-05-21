import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.models.bpr import BPRRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.pop import PopRecommender
from src.models.sasrec import SASRecRecommender
from src.utils.config import CATEGORIES, RESULTS_DIR, TOP_K
from src.utils.logger import get_logger

logger = get_logger("run_all")

TRADITIONAL_MODELS = ["pop", "item_cf", "bpr", "sasrec"]


def get_model(name: str, num_items: int):
    if name == "pop":
        return PopRecommender(num_items)
    elif name == "item_cf":
        return ItemCFRecommender(num_items)
    elif name == "bpr":
        return BPRRecommender(num_items)
    elif name == "sasrec":
        return SASRecRecommender(num_items)
    else:
        raise ValueError(f"Unknown model: {name}")


def run_category(category: str):
    logger.info(f"\n{'='*60}")
    logger.info(f"Category: {category}")
    logger.info(f"{'='*60}")

    dataset = AmazonDataset(category)
    train_data = dataset.get_train_sequences()
    test_data = dataset.get_eval_data("test")
    num_items = len(dataset.item_vocab)

    logger.info(f"Items: {num_items}, Train: {len(train_data)}, Test: {len(test_data)}")

    results = {}
    for model_name in TRADITIONAL_MODELS:
        logger.info(f"\n--- Training {model_name} ---")
        model = get_model(model_name, num_items)
        model.fit(train_data)

        evaluator = Evaluator(model, test_data, num_items, TOP_K)
        metrics = evaluator.evaluate()
        results[model.name] = metrics

    return {
        "category": category,
        "num_items": num_items,
        "num_train": len(train_data),
        "num_test": len(test_data),
        "results": results,
    }


def main():
    all_results = {}

    for category in CATEGORIES:
        result = run_category(category)
        all_results[category] = result

        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_file = os.path.join(RESULTS_DIR, f"{category}_all.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    for category, data in all_results.items():
        print(f"\n{category}:")
        print(f"  Items: {data['num_items']}, Train: {data['num_train']}, Test: {data['num_test']}")
        for model_name, metrics in data["results"].items():
            ndcg = metrics.get(f"NDCG@{TOP_K}", 0)
            hit = metrics.get(f"Hit@{TOP_K}", 0)
            mrr = metrics.get(f"MRR@{TOP_K}", 0)
            print(f"  {model_name:15s} NDCG@{TOP_K}={ndcg:.4f}  Hit@{TOP_K}={hit:.4f}  MRR@{TOP_K}={mrr:.4f}")

    summary_file = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
