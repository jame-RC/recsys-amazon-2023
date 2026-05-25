"""
BPR_Ultra v3 远程训练 — 简洁高效版

基于BPR_Advanced成功经验：简单架构+高学习率+均值池化
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.utils.config import TOP_K, RESULTS_DIR

CATEGORIES = ["Industrial_and_Scientific", "Musical_Instruments", "CDs_and_Vinyl"]


def get_dataset_stats(category):
    dataset = AmazonDataset(category)
    train_data = dataset.get_train_sequences()
    valid_data = dataset.get_eval_data("valid")
    test_data = dataset.get_eval_data("test")
    num_items = len(dataset.item_vocab)
    num_users = len(dataset.user_vocab)
    return {
        "train_data": train_data, "valid_data": valid_data, "test_data": test_data,
        "num_items": num_items, "num_users": num_users,
    }


def train_bpr_ultra(ds_info, category, dim=128, batch_size=16384,
                     neg_samples=10, lr=1e-3, weight_decay=1e-6,
                     num_epochs=100, patience=8, val_interval=2):
    from src.models.bpr_ultra import BPRUltraRecommender

    print(f"\n  >>> BPR_Ultra v3 [dim={dim}, bs={batch_size}, K={neg_samples}, "
           f"lr={lr}, wd={weight_decay}]")
    print(f"      Items={ds_info['num_items']}, Train={len(ds_info['train_data'])}")

    model = BPRUltraRecommender(
        num_items=ds_info["num_items"],
        dim=dim, lr=lr, weight_decay=weight_decay,
        num_epochs=num_epochs, patience=patience,
        neg_samples=neg_samples, batch_size=batch_size,
        val_interval=val_interval,
    )
    print(f"      Device: {model.device}")

    t0 = time.time()
    model.fit(ds_info["train_data"], category=category, valid_data=ds_info["valid_data"])
    train_time = time.time() - t0
    print(f"      Train time: {train_time:.1f}s")

    print(f"      Evaluating validation...", flush=True)
    valid_metrics = Evaluator(model, ds_info["valid_data"], ds_info["num_items"], TOP_K).evaluate()
    print(f"      Evaluating test...", flush=True)
    test_metrics = Evaluator(model, ds_info["test_data"], ds_info["num_items"], TOP_K).evaluate()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, f"{category}_BPR_Ultra_model.pt")
    torch.save(model.model.state_dict(), model_path)
    print(f"      Model saved to {model_path}")

    result = {
        "category": category, "model": "BPR_Ultra",
        "config": {"dim": dim, "batch_size": batch_size,
                    "neg_samples": neg_samples, "lr": lr, "weight_decay": weight_decay},
        "train_time_sec": round(train_time, 1),
        "valid": valid_metrics, "test": test_metrics,
    }
    result_path = os.path.join(RESULTS_DIR, f"{category}_bpr_ultra.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"      Results saved to {result_path}")
    return result, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--neg-samples", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-interval", type=int, default=2)
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"#  BPR_Ultra v3 训练配置")
    print(f"#  dim={args.dim}, bs={args.batch_size}, K={args.neg_samples}")
    print(f"#  lr={args.lr}, wd={args.weight_decay}")
    print(f"#  max_epochs={args.epochs}, patience={args.patience}, val_interval={args.val_interval}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"#    GPU {i}: {props.name}, {props.total_memory/1024**3:.1f}GB VRAM")
    print(f"{'#'*60}")

    categories = [args.category] if args.category else CATEGORIES
    all_results = {}

    for cat in categories:
        print(f"\n{'='*60}\n  Processing: {cat}\n{'='*60}")
        ds_info = get_dataset_stats(cat)
        num_samples = len(ds_info["train_data"])
        bs = min(args.batch_size, num_samples)

        result, model = train_bpr_ultra(
            ds_info, cat, dim=args.dim, batch_size=bs,
            neg_samples=args.neg_samples, lr=args.lr,
            weight_decay=args.weight_decay, num_epochs=args.epochs,
            patience=args.patience, val_interval=args.val_interval,
        )
        all_results[f"{cat}_BPR_Ultra"] = result
        del model
        torch.cuda.empty_cache()

    # 汇总对比
    print(f"\n\n{'='*60}")
    print(f"  BPR_Ultra v3 结果 vs BPR_Advanced")
    print(f"{'='*60}")
    print(f"  {'Category':25s} | {'Val NDCG':>9s} | {'Test NDCG':>10s} | {'Test Hit':>8s} | vs Adv")
    print(f"  {'-'*25} | {'-'*9} | {'-'*10} | {'-'*8} | {'-'*8}")

    for cat in categories:
        key = f"{cat}_BPR_Ultra"
        if key in all_results:
            r = all_results[key]
            vn = r["valid"]["NDCG@10"]
            tn = r["test"]["NDCG@10"]
            th = r["test"]["Hit@10"]

            # 对比BPR_Advanced
            adv_path = os.path.join(RESULTS_DIR, f"{cat}_bpr_advanced.json")
            if os.path.exists(adv_path):
                with open(adv_path) as f:
                    prev = json.load(f)
                pn = prev["test"]["NDCG@10"]
                imp = (tn - pn) / pn * 100
                print(f"  {cat:25s} | {vn:>9.4f} | {tn:>10.4f} | {th:>8.4f} | {imp:>+7.1f}%")
            else:
                print(f"  {cat:25s} | {vn:>9.4f} | {tn:>10.4f} | {th:>8.4f} |")

    # 保存汇总
    all_results_path = os.path.join(RESULTS_DIR, "all_results.json")
    if os.path.exists(all_results_path):
        with open(all_results_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        with open(all_results_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(all_results_path, "w") as f:
            json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}\n  DONE!\n{'='*60}")


if __name__ == "__main__":
    main()
