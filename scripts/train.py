import argparse
import atexit
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.models.bpr import BPRRecommender
from src.models.bpr_advanced import BPRAdvancedRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.pop import PopRecommender
from src.models.sasrec import SASRecRecommender
from src.utils.config import CATEGORIES, RESULTS_DIR, TOP_K
from src.utils.logger import get_logger
from train_monitor import TrainingMonitor

logger = get_logger("train")


def get_model(name: str, num_items: int):
    if name == "pop":
        return PopRecommender(num_items)
    elif name == "item_cf":
        return ItemCFRecommender(num_items)
    elif name == "bpr":
        return BPRRecommender(num_items)
    elif name == "bpr_advanced":
        return BPRAdvancedRecommender(num_items)
    elif name == "sasrec":
        return SASRecRecommender(num_items)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_and_evaluate(category: str, model_name: str, monitor: TrainingMonitor | None = None):
    logger.info(f"Loading data for {category}...")
    dataset = AmazonDataset(category)

    train_data = dataset.get_train_sequences()
    valid_data = dataset.get_eval_data("valid")
    test_data = dataset.get_eval_data("test")

    num_items = len(dataset.item_vocab)
    logger.info(f"Items: {num_items}, Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

    model = get_model(model_name, num_items)

    logger.info(f"Training {model.name}...")
    model.fit(train_data, monitor=monitor, category=category)

    logger.info("Evaluating on validation set...")
    valid_evaluator = Evaluator(model, valid_data, num_items, TOP_K)
    valid_metrics = valid_evaluator.evaluate()

    logger.info("Evaluating on test set...")
    test_evaluator = Evaluator(model, test_data, num_items, TOP_K)
    test_metrics = test_evaluator.evaluate()

    return {
        "category": category,
        "model": model.name,
        "valid": valid_metrics,
        "test": test_metrics,
    }


def main():
    # Fix Windows GBK encoding for emoji/log output
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Industrial_and_Scientific", choices=CATEGORIES)
    parser.add_argument("--model", default="sasrec", choices=["pop", "item_cf", "bpr", "bpr_advanced", "sasrec"])
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard server port")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable the web dashboard")
    args = parser.parse_args()

    # --- Auto-start training monitor dashboard ---
    if not args.no_dashboard:
        metrics_file = f"{args.category}_{args.model}_metrics.json"
        monitor = TrainingMonitor(metrics_file)
        monitor.on_start(category=args.category, model=args.model)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "train_monitor.server",
             "--port", str(args.dashboard_port),
             "--metrics", metrics_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env
        )
        atexit.register(lambda: server_proc.terminate() if server_proc.poll() is None else None)
        time.sleep(1)  # Let the server start
        print(f"\n  [Training Dashboard] http://localhost:{args.dashboard_port}\n")
    else:
        monitor = None
    # ---

    result = train_and_evaluate(args.category, args.model, monitor=monitor)

    if monitor:
        monitor.on_end(best_valid=result.get("valid", {}), best_test=result.get("test", {}))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, f"{args.category}_{args.model}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
