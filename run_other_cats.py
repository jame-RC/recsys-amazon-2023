"""用网格搜索最优配置跑剩余品类"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, numpy as np
from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.utils.config import TOP_K, RESULTS_DIR

# 最优配置（来自网格搜索）
BEST_CONFIG = {"dim": 128, "lr": 2e-3, "K": 5, "dropout": 0.05, "wd": 1e-6}
CATEGORIES = ["Musical_Instruments", "CDs_and_Vinyl"]
device = torch.device("cuda")

print(f"{'#'*60}\n# BPR_Ultra 全品类训练\n{'#'*60}")
for k, v in BEST_CONFIG.items():
    print(f"#   {k} = {v}")
print(f"#   GPU: {torch.cuda.get_device_properties(0).name}\n{'#'*60}")


class BPRModel(torch.nn.Module):
    def __init__(self, num_items, dim=128, dropout=0.0):
        super().__init__()
        self.item_emb = torch.nn.Embedding(num_items, dim, padding_idx=0)
        self.item_bias = torch.nn.Embedding(num_items, 1, padding_idx=0)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.dropout_l = torch.nn.Dropout(dropout) if dropout > 0 else None
        torch.nn.init.normal_(self.item_emb.weight, 0, 0.01)
        torch.nn.init.zeros_(self.item_bias.weight)

    def _drop(self, x):
        return self.dropout_l(x) if self.dropout_l and self.training else x

    def forward(self, user_repr, pos_items, neg_items):
        ps = (user_repr * self._drop(self.item_emb(pos_items))).sum(-1) + self.item_bias(pos_items).squeeze(-1)
        ns = (user_repr.unsqueeze(1) * self._drop(self.item_emb(neg_items))).sum(-1) + self.item_bias(neg_items).squeeze(-1)
        return ps, ns


class Rec:
    def __init__(self, num_items, model):
        self.num_items = num_items; self.model = model; self.device = device
        self.supports_user_ids = False
    def recommend_batch(self, histories, top_k=10, **kw):
        if not histories: return []
        self.model.eval()
        with torch.no_grad():
            trunc = [h[-200:] if h else [] for h in histories]
            ml = max((len(h) for h in trunc if h), default=0)
            if ml == 0: return [[] for _ in histories]
            B = len(histories); p = torch.zeros(B, ml, dtype=torch.long, device=device)
            for i, h in enumerate(trunc):
                if h: p[i, :len(h)] = torch.tensor(h, dtype=torch.long, device=device)
            e = self.model.item_emb(p); m = (p > 0).float().unsqueeze(-1)
            u = (e * m).sum(1) / m.sum(1).clamp(min=1)
            ai = torch.arange(1, self.num_items, device=device)
            sc = u @ self.model.item_emb(ai).T + self.model.item_bias(ai).squeeze(-1).unsqueeze(0)
            for i, h in enumerate(histories):
                for iid in h:
                    pos = iid-1
                    if 0 <= pos < sc.size(1): sc[i, pos] = -float("inf")
            tk = torch.topk(sc, top_k, dim=1)
            return [[ai[idx].item() for idx in tk.indices[i]] for i in range(B)]


for cat in CATEGORIES:
    print(f"\n{'='*60}\n  {cat}\n{'='*60}", flush=True)
    ds = AmazonDataset(cat)
    train = ds.get_train_sequences()
    valid = ds.get_eval_data("valid")
    test = ds.get_eval_data("test")
    ni = len(ds.item_vocab)
    print(f"  Items={ni}, Train={len(train)}, Valid={len(valid)}, Test={len(test)}", flush=True)

    # Build tensors
    histories, targets, max_len = [], [], 0
    for _, h, t in train:
        if not h: continue
        histories.append(h[-200:]); targets.append(t)
        max_len = max(max_len, len(h[-200:]))
    n = len(histories)
    pad = np.zeros((n, max_len), dtype=np.int64)
    lens_np = np.zeros(n, dtype=np.int64)
    for i, h in enumerate(histories):
        pad[i, :len(h)] = h; lens_np[i] = len(h)
    data = {
        "h": torch.tensor(pad, dtype=torch.long, device=device),
        "l": torch.tensor(lens_np, dtype=torch.long, device=device),
        "t": torch.tensor(targets, dtype=torch.long, device=device),
    }
    print(f"  Samples: {n}", flush=True)

    cfg = BEST_CONFIG
    model = BPRModel(ni, cfg["dim"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.999))
    bs, nb = 16384, (n + 16384 - 1) // 16384

    best_val, best_state, best_ep = -1.0, None, -1
    t0 = time.time()

    for ep in range(100):
        model.train(); tl = 0.0
        idx = torch.randperm(n, device=device)
        for bi in range(nb):
            b = idx[bi*bs:min((bi+1)*bs, n)]; B = b.size(0)
            bh, bl, bt = data["h"][b], data["l"][b], data["t"][b]
            emb = model.item_emb(bh)
            m = (bh > 0).float().unsqueeze(-1)
            lt = (emb * m).sum(1) / m.sum(1).clamp(min=1)
            st = emb[torch.arange(B, device=device), (bl-1).clamp(min=0)]
            a = torch.sigmoid(model.alpha)
            ur = a * st + (1-a) * lt
            neg = torch.randint(1, ni, (B, cfg["K"]), device=device)
            cf = (neg == bt.unsqueeze(1))
            if cf.any(): neg[cf] = torch.randint(1, ni, (int(cf.sum().item()),), device=device)
            ps, ns = model(ur, bt, neg)
            loss = -torch.log(torch.sigmoid(ps.unsqueeze(1) - ns) + 1e-10).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
            tl += loss.item() * B

        vm = Evaluator(Rec(ni, model), valid, ni, TOP_K).evaluate()
        vn = vm[f"NDCG@{TOP_K}"]
        mark = ""
        if vn > best_val:
            best_val, best_ep = vn, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = " ✨"
        print(f"  ep{ep:2d} loss={tl/n:.4f} val={vn:.4f}{mark}", flush=True)
        if ep - best_ep > 15:
            print(f"  Early stop @ ep{ep}", flush=True)
            break

    print(f"  Best val={best_val:.4f} @ ep{best_ep} [{time.time()-t0:.0f}s]", flush=True)

    # Test
    model.load_state_dict(best_state)
    tm = Evaluator(Rec(ni, model), test, ni, TOP_K).evaluate()
    print(f"  Test NDCG@10={tm['NDCG@10']:.4f}  Hit@10={tm['Hit@10']:.4f}", flush=True)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(best_state, f"{RESULTS_DIR}/{cat}_BPR_Ultra_Final_model.pt")
    res = {"category": cat, "model": "BPR_Ultra_Final", "config": cfg,
           "val_ndcg": best_val, "test": tm}
    with open(f"{RESULTS_DIR}/{cat}_bpr_ultra_final.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  Saved {cat}_BPR_Ultra_Final_model.pt", flush=True)

    # Compare with BPR_Advanced
    adv_path = f"{RESULTS_DIR}/{cat}_bpr_advanced.json"
    if os.path.exists(adv_path):
        with open(adv_path) as f:
            adv = json.load(f)
        imp = (tm["NDCG@10"] - adv["test"]["NDCG@10"]) / adv["test"]["NDCG@10"] * 100
        print(f"  vs BPR_Advanced: {adv['test']['NDCG@10']:.4f} → {tm['NDCG@10']:.4f} ({imp:+.1f}%)", flush=True)

    torch.cuda.empty_cache()

print(f"\n{'#'*60}\n  ✅ 全部完成！\n{'#'*60}", flush=True)
