"""BPR 网格搜索 — 系统化超参调优

搜索策略（顺序优化）：
Round 1: lr = [5e-4, 1e-3, 2e-3]  (固定 dim=64, K=5, wd=1e-6, dropout=0.0)
Round 2: dropout = [0.05, 0.1, 0.2]  (固定最优lr)
Round 3: K = [10]  (固定最优lr+dropout)
"""
import json, os, sys, time, subprocess, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.utils.config import TOP_K, RESULTS_DIR

CAT = "Industrial_and_Scientific"
RESULTS_FILE = "grid_search_results.json"

# 加载数据（只加载一次）
print(f"Loading {CAT} data...", flush=True)
ds = AmazonDataset(CAT)
train = ds.get_train_sequences()
valid = ds.get_eval_data("valid")
test = ds.get_eval_data("test")
num_items = len(ds.item_vocab)
print(f"Items={num_items}, Train={len(train)}, Valid={len(valid)}, Test={len(test)}", flush=True)


class BPRModel(torch.nn.Module):
    def __init__(self, num_items, dim, dropout=0.0):
        super().__init__()
        self.item_emb = torch.nn.Embedding(num_items, dim, padding_idx=0)
        self.item_bias = torch.nn.Embedding(num_items, 1, padding_idx=0)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        torch.nn.init.normal_(self.item_emb.weight, 0, 0.01)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_repr, pos_items, neg_items):
        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)
        if self.dropout is not None:
            pos_emb = self.dropout(pos_emb)
            neg_emb = self.dropout(neg_emb)
        pos_bias = self.item_bias(pos_items).squeeze(-1)
        neg_bias = self.item_bias(neg_items).squeeze(-1)
        pos_score = (user_repr * pos_emb).sum(dim=-1) + pos_bias
        neg_score = (user_repr.unsqueeze(1) * neg_emb).sum(dim=-1) + neg_bias
        return pos_score, neg_score


def build_data_tensors(train_data):
    """Build GPU tensors from training data"""
    histories, targets = [], []
    max_len = 0
    for _, h, t in train_data:
        if not h: continue
        histories.append(h[-200:])
        targets.append(t)
        max_len = max(max_len, len(h[-200:]))
    if not histories:
        return None, 0
    
    n = len(histories)
    pad = np.zeros((n, max_len), dtype=np.int64)
    lens = np.zeros(n, dtype=np.int64)
    for i, h in enumerate(histories):
        pad[i, :len(h)] = h
        lens[i] = len(h)
    return {
        "history": torch.tensor(pad, dtype=torch.long),
        "lens": torch.tensor(lens, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }, n


def train_and_eval(config, data_tensors, num_samples, device):
    """Train a single config and return best val NDCG"""
    dim = config["dim"]
    lr = config["lr"]
    wd = config.get("wd", 1e-6)
    K = config.get("K", 5)
    dropout = config.get("dropout", 0.0)
    max_epochs = config.get("epochs", 80)
    patience = config.get("patience", 10)
    batch_size = config.get("batch_size", 16384)

    config_id = f"dim{dim}_lr{lr}_wd{wd}_K{K}_drop{dropout}"
    print(f"\n{'='*50}", flush=True)
    print(f"  Config: {config_id}", flush=True)
    print(f"{'='*50}", flush=True)

    model = BPRModel(num_items, dim, dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))

    hist_t = data_tensors["history"].to(device)
    lens_t = data_tensors["lens"].to(device)
    tgt_t = data_tensors["targets"].to(device)

    n = num_samples
    bs = batch_size
    n_batches = (n + bs - 1) // bs

    best_val = -float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = -1
    results = []

    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        indices = torch.randperm(n, device=device)

        for bi in range(n_batches):
            b_start = bi * bs
            b_end = min(b_start + bs, n)
            idx = indices[b_start:b_end]
            B = b_end - b_start

            b_hist = hist_t[idx]
            b_lens = lens_t[idx]
            b_tgt = tgt_t[idx]

            # User representation
            emb = model.item_emb(b_hist)
            mask = (b_hist > 0).float().unsqueeze(-1)
            long_term = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            last = (b_lens - 1).clamp(min=0)
            short_term = emb[torch.arange(B, device=device), last]
            alpha = torch.sigmoid(model.alpha)
            user_repr = alpha * short_term + (1 - alpha) * long_term

            # Negative sampling
            neg = torch.randint(1, num_items, (B, K), device=device)
            conflict = (neg == b_tgt.unsqueeze(1))
            if conflict.any():
                neg[conflict] = torch.randint(1, num_items,
                                              (int(conflict.sum().item()),), device=device)

            pos_score, neg_score = model(user_repr, b_tgt, neg)
            loss = -torch.log(torch.sigmoid(pos_score.unsqueeze(1) - neg_score) + 1e-10).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            total_loss += loss.item() * B

        avg_loss = total_loss / n

        # Validate every epoch
        model.eval()
        val_metrics = Evaluator(model, valid, num_items, TOP_K).evaluate()
        val_ndcg = val_metrics[f"NDCG@{TOP_K}"]

        results.append({"epoch": epoch, "loss": round(avg_loss, 4), "val_ndcg": val_ndcg})

        if val_ndcg > best_val:
            best_val = val_ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} val={val_ndcg:.4f} ✨ BEST", flush=True)
        else:
            patience_counter += 1
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f} val={val_ndcg:.4f} (pat={patience_counter})", flush=True)
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}, best={best_val:.4f}@epoch{best_epoch}", flush=True)
                break

    train_time = time.time() - t0
    print(f"  Train time: {train_time:.1f}s", flush=True)
    print(f"  Best val NDCG@10: {best_val:.4f} at epoch {best_epoch}", flush=True)

    return {
        "config": config,
        "best_val_ndcg": best_val,
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time, 1),
        "total_params": total_params,
        "epochs_run": epoch + 1,
        "results": results,
    }, best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}, {props.total_memory/1024**3:.1f}GB", flush=True)

    # Build data tensors (CPU first to save GPU mem)
    data_tensors, num_samples = build_data_tensors(train)
    print(f"Training samples: {num_samples}", flush=True)

    # ===== Round 1: Learning Rate =====
    print(f"\n\n{'#'*60}", flush=True)
    print(f"# Round 1: Learning Rate Search", flush=True)
    print(f"{'#'*60}", flush=True)

    lr_values = [5e-4, 1e-3, 2e-3]
    round1_results = []

    for lr in lr_values:
        config = {"dim": 64, "lr": lr, "wd": 1e-6, "K": 5, "dropout": 0.0, "epochs": 80, "patience": 10}
        result, _ = train_and_eval(config, data_tensors, num_samples, device)
        round1_results.append(result)
        torch.cuda.empty_cache()

    # Round 1 summary
    print(f"\n{'='*50}", flush=True)
    print(f"  Round 1 Results: Learning Rate", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  {'lr':>10s} | {'Best Val NDCG':>14s} | {'Epoch':>6s} | {'Time':>8s}", flush=True)
    print(f"  {'-'*10} | {'-'*14} | {'-'*6} | {'-'*8}", flush=True)
    best_lr = None
    best_lr_val = -1
    for r in round1_results:
        c = r["config"]
        print(f"  {c['lr']:>10} | {r['best_val_ndcg']:>14.4f} | {r['best_epoch']:>6d} | {r['train_time_sec']:>8.1f}s", flush=True)
        if r["best_val_ndcg"] > best_lr_val:
            best_lr_val = r["best_val_ndcg"]
            best_lr = c["lr"]
    print(f"\n  Best LR: {best_lr} (val NDCG={best_lr_val:.4f})", flush=True)

    # ===== Round 2: Dropout =====
    print(f"\n\n{'#'*60}", flush=True)
    print(f"# Round 2: Dropout Search (lr={best_lr})", flush=True)
    print(f"{'#'*60}", flush=True)

    dropout_values = [0.05, 0.1, 0.2]
    round2_results = []

    for drop in dropout_values:
        config = {"dim": 64, "lr": best_lr, "wd": 1e-6, "K": 5, "dropout": drop, "epochs": 80, "patience": 10}
        result, _ = train_and_eval(config, data_tensors, num_samples, device)
        round2_results.append(result)
        torch.cuda.empty_cache()

    # Round 2 summary
    print(f"\n{'='*50}", flush=True)
    print(f"  Round 2 Results: Dropout", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  {'dropout':>10s} | {'Best Val NDCG':>14s} | {'Epoch':>6s} | {'Time':>8s}", flush=True)
    print(f"  {'-'*10} | {'-'*14} | {'-'*6} | {'-'*8}", flush=True)
    best_drop = 0.0
    best_drop_val = best_lr_val  # baseline (dropout=0.0)
    for r in round2_results:
        c = r["config"]
        print(f"  {c['dropout']:>10} | {r['best_val_ndcg']:>14.4f} | {r['best_epoch']:>6d} | {r['train_time_sec']:>8.1f}s", flush=True)
        if r["best_val_ndcg"] > best_drop_val:
            best_drop_val = r["best_val_ndcg"]
            best_drop = c["dropout"]
    print(f"\n  Best dropout: {best_drop} (val NDCG={best_drop_val:.4f})", flush=True)

    # ===== Round 3: Negative Samples K =====
    print(f"\n\n{'#'*60}", flush=True)
    print(f"# Round 3: K Search (lr={best_lr}, dropout={best_drop})", flush=True)
    print(f"{'#'*60}", flush=True)

    config = {"dim": 64, "lr": best_lr, "wd": 1e-6, "K": 10, "dropout": best_drop, "epochs": 80, "patience": 10}
    result_k10, _ = train_and_eval(config, data_tensors, num_samples, device)

    print(f"\n{'='*50}", flush=True)
    print(f"  K=10 vs K=5", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  K=5:  {best_drop_val:.4f}")
    print(f"  K=10: {result_k10['best_val_ndcg']:.4f}")

    best_config = None
    best_val = best_drop_val
    if result_k10["best_val_ndcg"] > best_val:
        best_val = result_k10["best_val_ndcg"]
        best_config = {"dim": 64, "lr": best_lr, "wd": 1e-6, "K": 10, "dropout": best_drop}
    else:
        best_config = {"dim": 64, "lr": best_lr, "wd": 1e-6, "K": 5, "dropout": best_drop}

    # ===== Final Summary =====
    print(f"\n\n{'#'*60}", flush=True)
    print(f"# 网格搜索完成!", flush=True)
    print(f"{'#'*60}", flush=True)
    print(f"  最优配置:", flush=True)
    print(f"    lr      = {best_config['lr']}", flush=True)
    print(f"    dim     = {best_config['dim']}", flush=True)
    print(f"    K       = {best_config['K']}", flush=True)
    print(f"    dropout = {best_config['dropout']}", flush=True)
    print(f"    wd      = {best_config.get('wd', 1e-6)}", flush=True)
    print(f"  Val NDCG@10 = {best_val:.4f}", flush=True)

    # Save all results
    all_results = {
        "round1_lr": round1_results,
        "round2_dropout": round2_results,
        "round3_K": {"K5": best_drop_val, "K10": result_k10["best_val_ndcg"]},
        "best_config": best_config,
        "best_val_ndcg": best_val,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  结果已保存到 {RESULTS_FILE}", flush=True)

    # ===== Final Training with Best Config =====
    print(f"\n\n{'#'*60}", flush=True)
    print(f"# 用最优配置训练最终模型（200 epochs）", flush=True)
    print(f"{'#'*60}", flush=True)

    final_config = {**best_config, "epochs": 200, "patience": 30}
    final_result, best_state = train_and_eval(final_config, data_tensors, num_samples, device)

    # Evaluate on test set
    print(f"\n  Evaluating on test set...", flush=True)
    
    # Restore best model
    final_model = BPRModel(num_items, final_config["dim"], final_config["dropout"]).to(device)
    final_model.load_state_dict(best_state)
    
    # Create a wrapper for evaluation
    class ModelWrapper:
        def __init__(self, model, device, num_items):
            self.model = model
            self.device = device
            self.num_items = num_items
            self.supports_user_ids = False
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

    wrapper = ModelWrapper(final_model, device, num_items)
    test_metrics = Evaluator(wrapper, test, num_items, TOP_K).evaluate()

    print(f"\n{'='*50}", flush=True)
    print(f"  ✅ 最终测试结果", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Test NDCG@10 = {test_metrics['NDCG@10']:.4f}", flush=True)
    print(f"  Test Hit@10  = {test_metrics['Hit@10']:.4f}", flush=True)
    print(f"  Test MRR@10  = {test_metrics['MRR@10']:.4f}", flush=True)

    # Compare with BPR_Advanced
    adv_path = f"{RESULTS_DIR}/{CAT}_bpr_advanced.json"
    if os.path.exists(adv_path):
        with open(adv_path) as f:
            adv = json.load(f)
        prev_ndcg = adv["test"]["NDCG@10"]
        imp = (test_metrics["NDCG@10"] - prev_ndcg) / prev_ndcg * 100
        print(f"\n  vs BPR_Advanced: {prev_ndcg:.4f} → {test_metrics['NDCG@10']:.4f} ({imp:+.1f}%)", flush=True)

    # Save final model
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(best_state, f"{RESULTS_DIR}/{CAT}_BPR_GridSearch_best_model.pt")

    final_output = {
        "category": CAT,
        "model": "BPR_GridSearch",
        "best_config": best_config,
        "grid_search_results": all_results,
        "test": test_metrics,
    }
    with open(f"{RESULTS_DIR}/{CAT}_bpr_gridsearch.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n  模型已保存到 {RESULTS_DIR}/{CAT}_BPR_GridSearch_best_model.pt", flush=True)
    print(f"\n{'#'*60}", flush=True)
    print(f"  ✅ 网格搜索完成！", flush=True)
    print(f"{'#'*60}", flush=True)


if __name__ == "__main__":
    main()
