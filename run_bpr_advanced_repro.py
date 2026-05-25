"""精确复现 BPR_Advanced 结果 — 使用原始默认参数"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.models.bpr_advanced import BPRAdvancedRecommender
from src.utils.config import TOP_K, RESULTS_DIR

CATEGORY = "Industrial_and_Scientific"

print(f"Loading data...", flush=True)
ds = AmazonDataset(CATEGORY)
train_data = ds.get_train_sequences()
valid_data = ds.get_eval_data("valid")
test_data = ds.get_eval_data("test")
num_items = len(ds.item_vocab)
print(f"Items={num_items}, Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}", flush=True)

# 使用BPR_Advanced的完全默认参数
model = BPRAdvancedRecommender(num_items)
print(f"Device={model.device}, dim={model.embedding_dim}, lr={model.lr}, "
      f"K={model.neg_samples}, wd={model.weight_decay}, patience={model.patience}", flush=True)

t0 = time.time()
model.fit(train_data, category=CATEGORY, valid_data=valid_data)
print(f"Train done in {time.time()-t0:.1f}s", flush=True)

print(f"Valid eval...", flush=True)
vm = Evaluator(model, valid_data, num_items, TOP_K).evaluate()

print(f"Test eval...", flush=True)
tm = Evaluator(model, test_data, num_items, TOP_K).evaluate()

print(f"\n=== RESULTS ===")
print(f"Valid NDCG@10={vm['NDCG@10']:.4f}, Hit@10={vm['Hit@10']:.4f}")
print(f"Test  NDCG@10={tm['NDCG@10']:.4f}, Hit@10={tm['Hit@10']:.4f}")

# Save
os.makedirs(RESULTS_DIR, exist_ok=True)
result = {"category": CATEGORY, "model": "BPR_Advanced_repro",
          "valid": vm, "test": tm, "train_time_sec": round(time.time()-t0, 1)}
with open(f"{RESULTS_DIR}/{CATEGORY}_bpr_advanced_repro.json", "w") as f:
    json.dump(result, f, indent=2)
import torch
torch.save(model.model.state_dict(), f"{RESULTS_DIR}/{CATEGORY}_BPR_Advanced_repro_model.pt")
print(f"Saved!", flush=True)
