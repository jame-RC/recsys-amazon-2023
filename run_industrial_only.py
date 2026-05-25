"""只跑 Industrial_and_Scientific 的 BPR_Ultra Final v2"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.utils.config import TOP_K, RESULTS_DIR
from train_bpr_final_v2 import BPRUltraFinal

CAT = "Industrial_and_Scientific"

print(f"Loading {CAT}...", flush=True)
ds = AmazonDataset(CAT)
train = ds.get_train_sequences()
valid = ds.get_eval_data("valid")
test = ds.get_eval_data("test")
num_items = len(ds.item_vocab)
print(f"Items={num_items}, Train={len(train)}, Valid={len(valid)}, Test={len(test)}", flush=True)

model = BPRUltraFinal(num_items, dim=128, num_epochs=200)
print(f"Device={model.device}", flush=True)

t0 = time.time()
model.fit(train, valid_data=valid)
train_time = time.time() - t0

print(f"\nEvaluating test set...", flush=True)
test_metrics = Evaluator(model, test, num_items, TOP_K).evaluate()

result = {
    "category": CAT,
    "model": "BPR_Ultra_Final_dim128",
    "config": {"dim": 128},
    "train_time_sec": round(train_time, 1),
    "test": test_metrics,
}

os.makedirs(RESULTS_DIR, exist_ok=True)
torch.save(model.model.state_dict(), f"{RESULTS_DIR}/{CAT}_BPR_Ultra_Final_v2_model.pt")
with open(f"{RESULTS_DIR}/{CAT}_bpr_ultra_final_v2.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n=== FINAL RESULT ===", flush=True)
print(f"Test NDCG@10={test_metrics['NDCG@10']:.4f}")
print(f"Test Hit@10={test_metrics['Hit@10']:.4f}")
print(f"Test MRR@10={test_metrics['MRR@10']:.4f}")

# 对比BPR_Advanced
adv_path = f"{RESULTS_DIR}/{CAT}_bpr_advanced.json"
if os.path.exists(adv_path):
    with open(adv_path) as f:
        adv = json.load(f)
    print(f"\nvs BPR_Advanced: {adv['test']['NDCG@10']:.4f} → {test_metrics['NDCG@10']:.4f} "
          f"({(test_metrics['NDCG@10'] - adv['test']['NDCG@10'])/adv['test']['NDCG@10']*100:+.1f}%)")

print(f"\nSaved!")
