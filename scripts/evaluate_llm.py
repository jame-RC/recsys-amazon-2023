import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.llm.api import LLMClient
from src.models.llm_hybrid import LLMHybridRecommender
from src.models.llm_rag import LLMRAGRecommender
from src.models.llm_ranker import LLMRankerRecommender
from src.models.sasrec import SASRecRecommender
from src.utils.config import CATEGORIES, RESULTS_DIR, TOP_K
from src.utils.logger import get_logger

logger = get_logger("evaluate_llm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Industrial_and_Scientific", choices=CATEGORIES)
    parser.add_argument("--model", default="sasrec")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--mode", default="zeroshot", choices=["zeroshot", "fewshot", "rag", "hybrid"])
    args = parser.parse_args()

    logger.info(f"Loading data for {category := args.category}...")
    dataset = AmazonDataset(category)

    train_data = dataset.get_train_sequences()
    test_data = dataset.get_eval_data("test")
    num_items = len(dataset.item_vocab)

    logger.info(f"Training base model ({args.model})...")
    if args.model == "sasrec":
        base_model = SASRecRecommender(num_items)
    else:
        raise ValueError(f"Unknown base model: {args.model}")
    base_model.fit(train_data)

    llm_client = LLMClient(provider=args.llm_provider, model=args.llm_model,
                            api_key=args.api_key, base_url=args.base_url)

    item_meta = {}
    for asin, meta in dataset.item_meta.items():
        item_meta[asin] = {**meta, "idx": dataset.item_vocab.encode(asin)}

    if args.mode == "zeroshot":
        rec = LLMRankerRecommender(num_items, base_model, llm_client, item_meta=item_meta, use_few_shot=False)
    elif args.mode == "fewshot":
        rec = LLMRankerRecommender(num_items, base_model, llm_client, item_meta=item_meta, use_few_shot=True)
    elif args.mode == "rag":
        rec = LLMRAGRecommender(num_items, base_model, llm_client, item_meta)
    elif args.mode == "hybrid":
        rec = LLMHybridRecommender(num_items, [base_model], llm_client, item_meta=item_meta)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logger.info(f"Evaluating {rec.name}...")
    evaluator = Evaluator(rec, test_data, num_items, TOP_K)
    metrics = evaluator.evaluate()

    result = {
        "category": category,
        "model": rec.name,
        "test": metrics,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, f"{category}_{rec.name}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
