import argparse
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

logger = get_logger("evaluate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Industrial_and_Scientific", choices=CATEGORIES)
    parser.add_argument("--model", default="sasrec", choices=["pop", "item_cf", "bpr", "sasrec"])
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    args = parser.parse_args()

    logger.info(f"Loading data for {args.category}...")
    dataset = AmazonDataset(args.category)

    train_data = dataset.get_train_sequences()
    eval_data = dataset.get_eval_data(args.split)
    num_items = len(dataset.item_vocab)

    logger.info(f"Items: {num_items}, Train: {len(train_data)}, Eval: {len(eval_data)}")

    if args.model == "pop":
        model = PopRecommender(num_items)
    elif args.model == "item_cf":
        model = ItemCFRecommender(num_items)
    elif args.model == "bpr":
        model = BPRRecommender(num_items)
    elif args.model == "sasrec":
        model = SASRecRecommender(num_items)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    logger.info(f"Training {model.name}...")
    model.fit(train_data)

    logger.info(f"Evaluating on {args.split} set...")
    evaluator = Evaluator(model, eval_data, num_items, TOP_K)
    metrics = evaluator.evaluate()

    print(f"\nResults for {model.name} on {args.category} ({args.split}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
