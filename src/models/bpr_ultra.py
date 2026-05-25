"""BPR_Ultra v5 — In-Batch BPR损失（难负样本+成对排序）

v4 诊断：in-batch交叉熵损失让所有物品竞争，但推荐是排序问题
v5 方案：In-batch负采样 + BPR pairwise损失
- 负样本：批次内其他用户的正样本（hard negatives）
- 损失：BPR pairwise (-log(sigmoid(pos - neg)))
- 这结合了in-batch的难负样本和BPR的成对排序优势
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.base import BaseRecommender
from src.utils.logger import get_logger

logger = get_logger("bpr_ultra")


class BPRUltraModel(nn.Module):
    def __init__(self, num_items: int, dim: int = 64):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, dim, padding_idx=0)
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.zeros_(self.item_bias.weight)


class BPRUltraRecommender(BaseRecommender):
    def __init__(
        self,
        num_items: int,
        dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        num_epochs: int = 100,
        patience: int = 8,
        batch_size: int = 16384,
        val_interval: int = 2,
        neg_samples: int = 0,  # 兼容训练脚本
    ):
        super().__init__(num_items)
        self.name = "BPR_Ultra"
        self.dim = dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_interval = val_interval
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, monitor=None, **kwargs):
        category = kwargs.get("category", "default")
        valid_data = kwargs.get("valid_data", None)

        self.model = BPRUltraModel(self.num_items, self.dim).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"BPR_Ultra v5 params: {total_params:,}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay, betas=(0.9, 0.999),
        )

        # GPU数据
        histories_list = []
        targets_list = []
        max_len = 0
        for _, history, target in train_data:
            if not history:
                continue
            histories_list.append(history[-200:])
            targets_list.append(target)
            max_len = max(max_len, len(history[-200:]))

        if not histories_list:
            return
        num_samples = len(histories_list)

        padded_hist = np.zeros((num_samples, max_len), dtype=np.int64)
        hist_lens = np.zeros(num_samples, dtype=np.int64)
        for i, h in enumerate(histories_list):
            padded_hist[i, :len(h)] = h
            hist_lens[i] = len(h)

        history_t = torch.tensor(padded_hist, dtype=torch.long, device=self.device)
        hist_lens_t = torch.tensor(hist_lens, dtype=torch.long, device=self.device)
        targets_t = torch.tensor(targets_list, dtype=torch.long, device=self.device)

        bs = self.batch_size
        n_batches = (num_samples + bs - 1) // bs

        logger.info(f"Train: {num_samples}samples, bs={bs}, {n_batches}batches/epoch, dim={self.dim}")

        best_val = -float("inf")
        best_state = None
        patience_counter = 0
        epoch_since_val = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            indices = torch.randperm(num_samples, device=self.device)

            for batch_i in range(n_batches):
                b_start = batch_i * bs
                b_end = min(b_start + bs, num_samples)
                batch_idx = indices[b_start:b_end]
                B = b_end - b_start

                b_history = history_t[batch_idx]
                b_lens = hist_lens_t[batch_idx]
                b_target = targets_t[batch_idx]

                # 用户表示
                emb = self.model.item_emb(b_history)
                mask = (b_history > 0).float().unsqueeze(-1)
                long_term = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                last_idx = (b_lens - 1).clamp(min=0)
                short_term = emb[torch.arange(B, device=self.device), last_idx]
                alpha = torch.sigmoid(self.model.alpha)
                user_repr = alpha * short_term + (1 - alpha) * long_term

                # 正样本得分 [B]
                pos_emb = self.model.item_emb(b_target)
                pos_bias = self.model.item_bias(b_target).squeeze(-1)
                pos_score = (user_repr * pos_emb).sum(dim=-1) + pos_bias

                # In-batch BPR损失
                # 所有正样本 [B, D] 作为其他用户的负样本
                all_item_emb = pos_emb  # [B, D]
                all_item_bias = pos_bias  # [B]

                # 每个用户对所有物品的分数 [B, B]
                neg_scores = user_repr @ all_item_emb.T + all_item_bias.unsqueeze(0)

                # 对角线是正样本，排除
                pos_scores_diag = pos_score.unsqueeze(1)  # [B, 1]
                eye_mask = ~torch.eye(B, dtype=torch.bool, device=self.device)  # [B, B]
                neg_scores_masked = neg_scores[eye_mask].view(B, B - 1)  # [B, B-1]

                # BPR loss: -log(sigmoid(pos - neg)) 对所有B-1个负样本取平均
                loss = -torch.log(torch.sigmoid(pos_scores_diag - neg_scores_masked) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optimizer.step()

                total_loss += loss.item() * B

            avg_loss = total_loss / num_samples

            # 验证
            val_ndcg = None
            epoch_since_val += 1
            if valid_data is not None and epoch_since_val >= self.val_interval:
                self.model.eval()
                from src.evaluation.evaluator import Evaluator
                from src.utils.config import TOP_K
                val_metrics = Evaluator(self, valid_data, self.num_items, TOP_K).evaluate()
                val_ndcg = val_metrics[f"NDCG@{TOP_K}"]
                epoch_since_val = 0

            if val_ndcg is not None:
                logger.info(f"Epoch {epoch:3d}: loss={avg_loss:.4f} val_NDCG@10={val_ndcg:.4f}")
                if val_ndcg > best_val:
                    best_val = val_ndcg
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    logger.info(f"  ✨ New best val NDCG@10={best_val:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping epoch {epoch}, best={best_val:.4f}")
                        break
            else:
                logger.info(f"Epoch {epoch:3d}: loss={avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"Restored checkpoint val NDCG@10={best_val:.4f}")
        torch.cuda.empty_cache()

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        if not history:
            return []
        self.model.eval()
        with torch.no_grad():
            hist_t = torch.tensor([history[-200:]], dtype=torch.long, device=self.device)
            emb = self.model.item_emb(hist_t)
            mask = (hist_t > 0).float().unsqueeze(-1)
            user_repr = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_items = torch.arange(1, self.num_items, device=self.device)
            scores = user_repr @ self.model.item_emb(all_items).T
            scores += self.model.item_bias(all_items).squeeze(-1).unsqueeze(0)
            for item_id in history:
                if 1 <= item_id < self.num_items:
                    scores[0, item_id - 1] = -float("inf")
            topk = torch.topk(scores[0], top_k).indices
            return [all_items[i].item() for i in topk]

    def recommend_batch(self, histories: List[List[int]], top_k: int = 10,
                         **kwargs) -> List[List[int]]:
        if not histories:
            return []
        self.model.eval()
        with torch.no_grad():
            trunc = [h[-200:] if h else [] for h in histories]
            max_len = max((len(h) for h in trunc if h), default=0)
            if max_len == 0:
                return [[] for _ in histories]
            B = len(histories)
            padded = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
            for i, h in enumerate(trunc):
                if h:
                    padded[i, :len(h)] = torch.tensor(h, dtype=torch.long, device=self.device)
            emb = self.model.item_emb(padded)
            mask = (padded > 0).float().unsqueeze(-1)
            user_repr = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_items = torch.arange(1, self.num_items, device=self.device)
            all_scores = user_repr @ self.model.item_emb(all_items).T
            all_scores += self.model.item_bias(all_items).squeeze(-1).unsqueeze(0)
            for i, h in enumerate(histories):
                for item_id in h:
                    pos = item_id - 1
                    if 0 <= pos < all_scores.size(1):
                        all_scores[i, pos] = -float("inf")
            topk = torch.topk(all_scores, top_k, dim=1)
            return [[all_items[idx].item() for idx in topk.indices[i]] for i in range(B)]
