import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_monitor import TrainingMonitor
from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.models.sasrec import SASRecRecommender
from src.utils.config import TOP_K, RESULTS_DIR

# Fix: generate_square_subsequent_mask for PyTorch 2.5 - already works
# Use optimizations for RTX 2060 6GB
import src.utils.config as cfg
cfg.BATCH_SIZE = 1024       # Larger batch for GPU efficiency
cfg.NEG_SAMPLES = 50         # Fewer negatives for speed

monitor = TrainingMonitor("sasrec_metrics.json")
monitor.on_start(category="Industrial_and_Scientific", model="SASRec", gpu=torch.cuda.get_device_name(0))

cat = "Industrial_and_Scientific"
dataset = AmazonDataset(cat)
train_data = dataset.get_train_sequences()
valid_data = dataset.get_eval_data("valid")
test_data = dataset.get_eval_data("test")
num_items = len(dataset.item_vocab)
print(f"Items: {num_items}, Train: {len(train_data)}, Valid: {len(valid_data)}", flush=True)

# SASRec with GPU-optimized settings
model = SASRecRecommender(
    num_items, 
    embedding_dim=64,
    num_epochs=30,
    lr=3e-4,
    batch_size=1024,
    neg_samples=50
)
print(f"Device: {model.device}", flush=True)

t0 = time.time()
model.fit(train_data, monitor=monitor, category=cat)
print(f"Training: {time.time()-t0:.1f}s", flush=True)

print("Evaluating valid...", flush=True)
t0 = time.time()
vm = Evaluator(model, valid_data, num_items, TOP_K).evaluate()
print(f"Valid: NDCG@10={vm['NDCG@10']:.4f} ({time.time()-t0:.1f}s)", flush=True)

print("Evaluating test...", flush=True)
t0 = time.time()
tm = Evaluator(model, test_data, num_items, TOP_K).evaluate()
print(f"Test: NDCG@10={tm['NDCG@10']:.4f} ({time.time()-t0:.1f}s)", flush=True)

# Save
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.save(model.model.state_dict(), f"{RESULTS_DIR}/Industrial_and_Scientific_SASRec_model.pt")
result = {"category": cat, "model": "SASRec", "valid": vm, "test": tm}
import json
json.dump(result, open(f"{RESULTS_DIR}/Industrial_and_Scientific_sasrec.json", "w"), indent=2)
print("Saved!", flush=True)
monitor.on_end(best_valid=vm, best_test=tm)
