"""快速网格搜索 v2 — 每个配置25轮快速对比"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.utils.config import TOP_K, RESULTS_DIR

CAT = "Industrial_and_Scientific"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Load data once
ds = AmazonDataset(CAT)
train = ds.get_train_sequences()
valid = ds.get_eval_data("valid")
test = ds.get_eval_data("test")
num_items = len(ds.item_vocab)
print(f"Items={num_items}, Train={len(train)}, Valid={len(valid)}", flush=True)

# Build GPU tensors
histories, targets = [], []
max_len = 0
for _, h, t in train:
    if not h: continue
    histories.append(h[-200:]); targets.append(t)
    max_len = max(max_len, len(h[-200:]))
n = len(histories)
pad = np.zeros((n, max_len), dtype=np.int64)
lens = np.zeros(n, dtype=np.int64)
for i, h in enumerate(histories):
    pad[i, :len(h)] = h; lens[i] = len(h)
data = {
    "history": torch.tensor(pad, dtype=torch.long, device=device),
    "lens": torch.tensor(lens, dtype=torch.long, device=device),
    "targets": torch.tensor(targets, dtype=torch.long, device=device),
}
print(f"Samples: {n}", flush=True)


class BPRModel(torch.nn.Module):
    def __init__(self, num_items, dim=64, dropout=0.0):
        super().__init__()
        self.item_emb = torch.nn.Embedding(num_items, dim, padding_idx=0)
        self.item_bias = torch.nn.Embedding(num_items, 1, padding_idx=0)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.dropout_l = torch.nn.Dropout(dropout) if dropout > 0 else None
        torch.nn.init.normal_(self.item_emb.weight, 0, 0.01)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_repr, pos_items, neg_items):
        pos_emb = self._apply_dropout(self.item_emb(pos_items))
        neg_emb = self._apply_dropout(self.item_emb(neg_items))
        pos_bias = self.item_bias(pos_items).squeeze(-1)
        neg_bias = self.item_bias(neg_items).squeeze(-1)
        pos_score = (user_repr * pos_emb).sum(dim=-1) + pos_bias
        neg_score = (user_repr.unsqueeze(1) * neg_emb).sum(dim=-1) + neg_bias
        return pos_score, neg_score

    def _apply_dropout(self, x):
        if self.dropout_l is not None and self.training:
            return self.dropout_l(x)
        return x


class BPRRecommender:
    """BPR推荐器（统一接口）"""
    def __init__(self, num_items, dim=64, dropout=0.0):
        self.num_items = num_items
        self.device = device
        self.model = BPRModel(num_items, dim, dropout).to(device)
        self.supports_user_ids = False

    def recommend_batch(self, histories, top_k=10, **kwargs):
        if not histories: return []
        self.model.eval()
        with torch.no_grad():
            trunc = [h[-200:] if h else [] for h in histories]
            ml = max((len(h) for h in trunc if h), default=0)
            if ml == 0: return [[] for _ in histories]
            B = len(histories)
            padded = torch.zeros(B, ml, dtype=torch.long, device=device)
            for i, h in enumerate(trunc):
                if h: padded[i, :len(h)] = torch.tensor(h, dtype=torch.long, device=device)
            emb = self.model.item_emb(padded)
            mask = (padded > 0).float().unsqueeze(-1)
            ur = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            ai = torch.arange(1, self.num_items, device=device)
            sc = ur @ self.model.item_emb(ai).T + self.model.item_bias(ai).squeeze(-1).unsqueeze(0)
            for i, h in enumerate(histories):
                for iid in h:
                    pos = iid - 1
                    if 0 <= pos < sc.size(1): sc[i, pos] = -float("inf")
            topk = torch.topk(sc, top_k, dim=1)
            return [[ai[idx].item() for idx in topk.indices[i]] for i in range(B)]

    def recommend(self, history, top_k=10):
        return self.recommend_batch([history], top_k)[0]


def train_config(dim, lr, K, dropout, wd=1e-6, max_epochs=25, name=""):
    """Train one config and return best val NDCG"""
    tag = f"dim{dim}_lr{lr}_K{K}_drop{dropout}"
    print(f"\n{'='*50}\n  {name or tag}\n{'='*50}", flush=True)

    rec = BPRRecommender(num_items, dim, dropout)
    model = rec.model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    bs = 16384
    nb = (n + bs - 1) // bs

    best_val = -1.0
    best_ep = -1
    t0 = time.time()

    for ep in range(max_epochs):
        model.train()
        loss_total = 0.0
        idx = torch.randperm(n, device=device)
        for bi in range(nb):
            b = idx[bi*bs:min((bi+1)*bs, n)]
            B = b.size(0)
            bh = data["history"][b]
            bl = data["lens"][b]
            bt = data["targets"][b]

            emb = model.item_emb(bh)
            mask = (bh > 0).float().unsqueeze(-1)
            lt = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            st = emb[torch.arange(B, device=device), (bl - 1).clamp(min=0)]
            alpha = torch.sigmoid(model.alpha)
            ur = alpha * st + (1 - alpha) * lt

            neg = torch.randint(1, num_items, (B, K), device=device)
            conflict = (neg == bt.unsqueeze(1))
            if conflict.any():
                neg[conflict] = torch.randint(1, num_items, (int(conflict.sum().item()),), device=device)

            ps, ns = model(ur, bt, neg)
            loss = -torch.log(torch.sigmoid(ps.unsqueeze(1) - ns) + 1e-10).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            loss_total += loss.item() * B

        # Validate
        vm = Evaluator(rec, valid, num_items, TOP_K).evaluate()
        vn = vm[f"NDCG@{TOP_K}"]
        marker = " ✨" if vn > best_val else ""
        if vn > best_val:
            best_val = vn
            best_ep = ep
        print(f"  ep{ep:2d} loss={loss_total/n:.4f} val={vn:.4f}{marker}", flush=True)

    tm = time.time() - t0
    print(f"  => Best: {best_val:.4f} @ ep{best_ep} [{tm:.0f}s]", flush=True)
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "config": {"dim": dim, "lr": lr, "K": K, "dropout": dropout, "wd": wd},
        "best_val": best_val,
        "best_ep": best_ep,
        "time": round(tm, 1),
    }


# ===== GRID SEARCH =====
all_results = []

# Round 1: Learning Rate (固定 dim=64, K=5, dropout=0)
print(f"\n\n{'#'*60}\n# Round 1: Learning Rate\n{'#'*60}", flush=True)
for lr in [5e-4, 1e-3, 2e-3]:
    r = train_config(64, lr, 5, 0.0, name=f"LR={lr}")
    all_results.append(r)

best = max(all_results, key=lambda x: x["best_val"])
best_lr = best["config"]["lr"]
print(f"\n>>> Best LR = {best_lr} (val={best['best_val']:.4f})", flush=True)

# Round 2: Dropout (用最优LR)
print(f"\n\n{'#'*60}\n# Round 2: Dropout\n{'#'*60}", flush=True)
for drop in [0.05, 0.1]:
    r = train_config(64, best_lr, 5, drop, name=f"drop={drop}")
    all_results.append(r)

best_drop = max([r for r in all_results if r["config"]["lr"] == best_lr and r["config"]["K"] == 5],
                key=lambda x: x["best_val"])
best_dropout = best_drop["config"]["dropout"]
print(f"\n>>> Best dropout = {best_dropout} (val={best_drop['best_val']:.4f})", flush=True)

# Round 3: K=10 vs K=5
print(f"\n\n{'#'*60}\n# Round 3: K=10\n{'#'*60}", flush=True)
r_k10 = train_config(64, best_lr, 10, best_dropout, name="K=10")
all_results.append(r_k10)

best_K_config = max(
    [r for r in all_results if r["config"]["lr"] == best_lr and r["config"]["dropout"] == best_dropout],
    key=lambda x: x["best_val"]
)
best_K = best_K_config["config"]["K"]
print(f"\n>>> Best K = {best_K} (val={best_K_config['best_val']:.4f})", flush=True)

# Round 4: dim=128 vs dim=64
print(f"\n\n{'#'*60}\n# Round 4: dim=128\n{'#'*60}", flush=True)
r_dim128 = train_config(128, best_lr, best_K, best_dropout, name="dim=128")
all_results.append(r_dim128)

# Overall best
all_results.sort(key=lambda x: x["best_val"], reverse=True)
best = all_results[0]
print(f"\n\n{'='*60}", flush=True)
print(f"  快速搜索完成！", flush=True)
print(f"{'='*60}", flush=True)
print(f"  排名:", flush=True)
for i, r in enumerate(all_results):
    cfg = r["config"]
    print(f"  {i+1}. dim={cfg['dim']} lr={cfg['lr']} K={cfg['K']} drop={cfg['dropout']} "
          f"→ val={r['best_val']:.4f} @ep{r['best_ep']}", flush=True)

best_cfg = best["config"]
print(f"\n  最优配置: dim={best_cfg['dim']}, lr={best_cfg['lr']}, "
      f"K={best_cfg['K']}, dropout={best_cfg['dropout']}", flush=True)

# Save screening
with open("grid_search_early.json", "w") as f:
    json.dump({"results": all_results, "best": best}, f, indent=2)

# ===== FINAL TRAINING =====
print(f"\n\n{'#'*60}", flush=True)
print(f"  最终训练: 最优配置 × 100 epochs", flush=True)
print(f"{'#'*60}", flush=True)

rec = BPRRecommender(num_items, best_cfg["dim"], best_cfg["dropout"])
model = rec.model
opt = torch.optim.AdamW(model.parameters(), lr=best_cfg["lr"],
                         weight_decay=best_cfg["wd"], betas=(0.9, 0.999))
bs = 16384
nb = (n + bs - 1) // bs

best_val = -1.0
best_state = None
best_ep = -1
t0 = time.time()

for ep in range(100):
    model.train()
    loss_total = 0.0
    idx = torch.randperm(n, device=device)
    for bi in range(nb):
        b = idx[bi*bs:min((bi+1)*bs, n)]
        B = b.size(0)
        bh, bl, bt = data["history"][b], data["lens"][b], data["targets"][b]

        emb = model.item_emb(bh)
        mask = (bh > 0).float().unsqueeze(-1)
        lt = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        st = emb[torch.arange(B, device=device), (bl - 1).clamp(min=0)]
        alpha = torch.sigmoid(model.alpha)
        ur = alpha * st + (1 - alpha) * lt

        neg = torch.randint(1, num_items, (B, best_cfg["K"]), device=device)
        conflict = (neg == bt.unsqueeze(1))
        if conflict.any():
            neg[conflict] = torch.randint(1, num_items, (int(conflict.sum().item()),), device=device)

        ps, ns = model(ur, bt, neg)
        loss = -torch.log(torch.sigmoid(ps.unsqueeze(1) - ns) + 1e-10).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        loss_total += loss.item() * B

    # Validate
    vm = Evaluator(rec, valid, num_items, TOP_K).evaluate()
    vn = vm[f"NDCG@{TOP_K}"]

    marker = ""
    if vn > best_val:
        best_val = vn
        best_ep = ep
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        marker = " ✨BEST"
    print(f"  ep{ep:2d} loss={loss_total/n:.4f} val={vn:.4f}{marker}", flush=True)

    if ep - best_ep > 15:
        print(f"  Early stop (no improvement for 15 epochs)", flush=True)
        break

train_time = time.time() - t0
print(f"\n  Best val NDCG@10 = {best_val:.4f} @ epoch {best_ep} [{train_time:.0f}s]", flush=True)

# Test evaluation
print(f"\n  Evaluating test set...", flush=True)
model.load_state_dict(best_state)
rec.model = model
test_metrics = Evaluator(rec, test, num_items, TOP_K).evaluate()
test_ndcg = test_metrics["NDCG@10"]

print(f"\n{'='*60}", flush=True)
print(f"  ✅ 最终结果", flush=True)
print(f"{'='*60}", flush=True)
print(f"  最优配置: dim={best_cfg['dim']}, lr={best_cfg['lr']}, K={best_cfg['K']}, "
      f"dropout={best_cfg['dropout']}", flush=True)
print(f"  Test NDCG@10 = {test_ndcg:.4f}", flush=True)
print(f"  Test Hit@10  = {test_metrics['Hit@10']:.4f}", flush=True)
print(f"  Test MRR@10  = {test_metrics['MRR@10']:.4f}", flush=True)

# Compare with BPR_Advanced
adv_path = f"{RESULTS_DIR}/{CAT}_bpr_advanced.json"
if os.path.exists(adv_path):
    with open(adv_path) as f:
        adv = json.load(f)
    prev = adv["test"]["NDCG@10"]
    imp = (test_ndcg - prev) / prev * 100
    print(f"  vs BPR_Advanced: {prev:.4f} → {test_ndcg:.4f} ({imp:+.1f}%)", flush=True)

# Save final model
os.makedirs(RESULTS_DIR, exist_ok=True)
torch.save(best_state, f"{RESULTS_DIR}/{CAT}_BPR_GridSearch_best_model.pt")
final_output = {
    "category": CAT, "model": "BPR_GridSearch",
    "best_config": best_cfg, "screening_results": all_results,
    "test": test_metrics,
}
with open(f"{RESULTS_DIR}/{CAT}_bpr_gridsearch.json", "w") as f:
    json.dump(final_output, f, indent=2)
print(f"\n  模型已保存到 {RESULTS_DIR}/{CAT}_BPR_GridSearch_best_model.pt", flush=True)
print(f"\n  ✅ DONE!", flush=True)
