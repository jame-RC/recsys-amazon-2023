"""BPR_Ultra Final — 基于BPR_Advanced验证的配置 + dim=128

已验证的基线：dim=64, K=5, lr=1e-3, wd=1e-6 → Val 0.0265
改进：dim=128 (更多容量) + K=5 (稳定负采样)
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.utils.config import TOP_K, RESULTS_DIR

CATEGORIES = ["Industrial_and_Scientific", "Musical_Instruments", "CDs_and_Vinyl"]


class BPRUltraModel(torch.nn.Module):
    def __init__(self, num_items, dim=128):
        super().__init__()
        self.item_emb = torch.nn.Embedding(num_items, dim, padding_idx=0)
        self.item_bias = torch.nn.Embedding(num_items, 1, padding_idx=0)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        torch.nn.init.normal_(self.item_emb.weight, 0, 0.01)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_repr, pos_items, neg_items):
        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)
        pos_bias = self.item_bias(pos_items).squeeze(-1)
        neg_bias = self.item_bias(neg_items).squeeze(-1)
        pos_score = (user_repr * pos_emb).sum(dim=-1) + pos_bias
        neg_score = (user_repr.unsqueeze(1) * neg_emb).sum(dim=-1) + neg_bias
        return pos_score, neg_score


class BPRUltraFinal:
    """BPR_Ultra Final — 完全基于验证配置"""
    name = "BPR_Ultra_Final"
    
    def __init__(self, num_items, dim=128, lr=1e-3, wd=1e-6,
                 num_epochs=100, patience=8, neg_samples=5,
                 batch_size=16384):
        self.num_items = num_items
        self.dim = dim
        self.lr = lr
        self.wd = wd
        self.num_epochs = num_epochs
        self.patience = patience
        self.neg_samples = neg_samples
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.supports_user_ids = False

    def fit(self, train_data, valid_data=None):
        self.model = BPRUltraModel(self.num_items, self.dim).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Params: {total_params:,}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.wd, betas=(0.9, 0.999),
        )

        # GPU data
        histories, targets = [], []
        max_len = 0
        for _, h, t in train_data:
            if not h: continue
            histories.append(h[-200:])
            targets.append(t)
            max_len = max(max_len, len(h[-200:]))
        if not histories: return

        n = len(histories)
        pad = np.zeros((n, max_len), dtype=np.int64)
        lens = np.zeros(n, dtype=np.int64)
        for i, h in enumerate(histories):
            pad[i, :len(h)] = h
            lens[i] = len(h)

        hist_t = torch.tensor(pad, dtype=torch.long, device=self.device)
        lens_t = torch.tensor(lens, dtype=torch.long, device=self.device)
        tgt_t = torch.tensor(targets, dtype=torch.long, device=self.device)

        bs = self.batch_size
        n_batches = (n + bs - 1) // bs
        K = self.neg_samples

        print(f"  Train: {n}samples, bs={bs}, {n_batches}batches, K={K}, dim={self.dim}")

        best_val = -float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            indices = torch.randperm(n, device=self.device)

            for bi in range(n_batches):
                b_start = bi * bs
                b_end = min(b_start + bs, n)
                idx = indices[b_start:b_end]
                B = b_end - b_start

                b_hist = hist_t[idx]
                b_lens = lens_t[idx]
                b_tgt = tgt_t[idx]

                # User representation
                emb = self.model.item_emb(b_hist)
                mask = (b_hist > 0).float().unsqueeze(-1)
                long_term = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                last = (b_lens - 1).clamp(min=0)
                short_term = emb[torch.arange(B, device=self.device), last]
                alpha = torch.sigmoid(self.model.alpha)
                user_repr = alpha * short_term + (1 - alpha) * long_term

                # K random negatives
                neg = torch.randint(1, self.num_items, (B, K), device=self.device)
                conflict = (neg == b_tgt.unsqueeze(1))
                if conflict.any():
                    neg[conflict] = torch.randint(1, self.num_items,
                                                  (int(conflict.sum().item()),), device=self.device)

                pos_score, neg_score = self.model(user_repr, b_tgt, neg)
                loss = -torch.log(torch.sigmoid(pos_score.unsqueeze(1) - neg_score) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optimizer.step()
                total_loss += loss.item() * B

            avg_loss = total_loss / n

            # Every epoch validation
            if valid_data is not None:
                self.model.eval()
                val_metrics = Evaluator(self, valid_data, self.num_items, TOP_K).evaluate()
                val_ndcg = val_metrics[f"NDCG@{TOP_K}"]
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} val_NDCG@10={val_ndcg:.4f}")

                if val_ndcg > best_val:
                    best_val = val_ndcg
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    print(f"    ✨ New best!")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"  Early stopping at epoch {epoch}, best={best_val:.4f}")
                        break
            else:
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  Restored best checkpoint val NDCG@10={best_val:.4f}")
        torch.cuda.empty_cache()

    def recommend(self, history, top_k=10):
        if not history: return []
        self.model.eval()
        with torch.no_grad():
            ht = torch.tensor([history[-200:]], dtype=torch.long, device=self.device)
            emb = self.model.item_emb(ht)
            mask = (ht > 0).float().unsqueeze(-1)
            ur = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_i = torch.arange(1, self.num_items, device=self.device)
            scores = ur @ self.model.item_emb(all_i).T
            scores += self.model.item_bias(all_i).squeeze(-1).unsqueeze(0)
            for iid in history:
                if 1 <= iid < self.num_items:
                    scores[0, iid-1] = -float("inf")
            topk = torch.topk(scores[0], top_k).indices
            return [all_i[i].item() for i in topk]

    def recommend_batch(self, histories, top_k=10, **kwargs):
        if not histories: return []
        self.model.eval()
        with torch.no_grad():
            trunc = [h[-200:] if h else [] for h in histories]
            ml = max((len(h) for h in trunc if h), default=0)
            if ml == 0: return [[] for _ in histories]
            B = len(histories)
            pad = torch.zeros(B, ml, dtype=torch.long, device=self.device)
            for i, h in enumerate(trunc):
                if h: pad[i, :len(h)] = torch.tensor(h, dtype=torch.long, device=self.device)
            emb = self.model.item_emb(pad)
            mask = (pad > 0).float().unsqueeze(-1)
            ur = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_i = torch.arange(1, self.num_items, device=self.device)
            sc = ur @ self.model.item_emb(all_i).T
            sc += self.model.item_bias(all_i).squeeze(-1).unsqueeze(0)
            for i, h in enumerate(histories):
                for iid in h:
                    pos = iid - 1
                    if 0 <= pos < sc.size(1): sc[i, pos] = -float("inf")
            topk = torch.topk(sc, top_k, dim=1)
            return [[all_i[idx].item() for idx in topk.indices[i]] for i in range(B)]


def train_category(cat, dim=128):
    print(f"\n{'='*60}")
    print(f"  {cat} (dim={dim})")
    print(f"{'='*60}")
    
    ds = AmazonDataset(cat)
    train = ds.get_train_sequences()
    valid = ds.get_eval_data("valid")
    test = ds.get_eval_data("test")
    num_items = len(ds.item_vocab)
    print(f"  Items={num_items}, Train={len(train)}, Valid={len(valid)}, Test={len(test)}")

    model = BPRUltraFinal(num_items, dim=dim)
    print(f"  Device={model.device}")

    t0 = time.time()
    model.fit(train, valid_data=valid)
    train_time = time.time() - t0

    print(f"  Test evaluation...", flush=True)
    test_metrics = Evaluator(model, test, num_items, TOP_K).evaluate()

    result = {
        "category": cat, "model": f"BPR_Ultra_Final_dim{dim}",
        "config": {"dim": dim},
        "train_time_sec": round(train_time, 1),
        "test": test_metrics,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.model.state_dict(), f"{RESULTS_DIR}/{cat}_BPR_Ultra_Final_model.pt")
    with open(f"{RESULTS_DIR}/{cat}_bpr_ultra_final.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved!")

    return result, model


def main():
    print(f"\n{'#'*60}")
    print(f"#  BPR_Ultra Final — 全品类训练")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"#    GPU {i}: {props.name}, {props.total_memory/1024**3:.1f}GB VRAM")
    print(f"{'#'*60}")

    results = {}
    for cat in CATEGORIES:
        result, model = train_category(cat, dim=128)
        results[f"{cat}_BPR_Ultra_Final"] = result
        del model
        torch.cuda.empty_cache()

    print(f"\n\n{'='*60}")
    print(f"  BPR_Ultra Final 结果汇总 (dim=128, K=5)")
    print(f"{'='*60}")
    print(f"  {'Category':25s} | {'Test NDCG@10':>13s} | {'Test Hit@10':>12s}")
    print(f"  {'-'*25} | {'-'*13} | {'-'*12}")
    for cat in CATEGORIES:
        key = f"{cat}_BPR_Ultra_Final"
        if key in results:
            r = results[key]
            print(f"  {cat:25s} | {r['test']['NDCG@10']:>13.4f} | {r['test']['Hit@10']:>12.4f}")

    # 对比BPR_Advanced
    print(f"\n  对比 BPR_Advanced (dim=64):")
    for cat in CATEGORIES:
        key = f"{cat}_BPR_Ultra_Final"
        adv_path = os.path.join(RESULTS_DIR, f"{cat}_bpr_advanced.json")
        if key in results and os.path.exists(adv_path):
            with open(adv_path) as f:
                adv = json.load(f)
            cur = results[key]["test"]["NDCG@10"]
            prev = adv["test"]["NDCG@10"]
            imp = (cur - prev) / prev * 100
            print(f"  {cat:25s}: {prev:.4f} → {cur:.4f} ({imp:+.1f}%)")

    # 保存汇总
    all_path = os.path.join(RESULTS_DIR, "all_results.json")
    if os.path.exists(all_path):
        with open(all_path) as f:
            existing = json.load(f)
        existing.update(results)
        with open(all_path, "w") as f:
            json.dump(existing, f, indent=2)
    print(f"\n{'='*60}\n  DONE!\n{'='*60}")


if __name__ == "__main__":
    main()
