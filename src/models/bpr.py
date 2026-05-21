from typing import List

import torch
import torch.nn as nn

from src.models.base import BaseRecommender
from src.utils.config import EMBEDDING_DIM, LEARNING_RATE, NUM_EPOCHS, PATIENCE
from src.utils.logger import get_logger

logger = get_logger("bpr")


class BPRModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)

    def forward(self, user_repr, pos_items, neg_items):
        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)
        pos_score = (user_repr * pos_emb).sum(dim=-1)
        neg_score = (user_repr * neg_emb).sum(dim=-1)
        return pos_score, neg_score


class BPRRecommender(BaseRecommender):
    def __init__(self, num_items: int, embedding_dim: int = EMBEDDING_DIM, lr: float = LEARNING_RATE,
                 num_epochs: int = NUM_EPOCHS, patience: int = PATIENCE):
        super().__init__(num_items)
        self.name = "BPR"
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, monitor=None):
        self.model = BPRModel(self.num_items, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Build user -> items mapping
        user_items = {}
        for user_idx, history, target in train_data:
            if user_idx not in user_items:
                user_items[user_idx] = set()
            user_items[user_idx].update(history)
            user_items[user_idx].add(target)

        user_item_lists = [sorted(items) for items in user_items.values() if len(items) >= 2]
        if not user_item_lists:
            return

        num_users = len(user_item_lists)
        # Pre-compute per-user: [item_count, cumulative_offset]
        offsets = [0]
        for items in user_item_lists:
            offsets.append(offsets[-1] + len(items))
        offsets_t = torch.tensor(offsets, device=self.device)
        total_triples = offsets[-1]

        # Flatten all items into one tensor for fast indexing
        all_items_flat = torch.tensor(
            [item for items in user_item_lists for item in items],
            dtype=torch.long, device=self.device
        )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0

            # --- Pre-compute user representations ---
            with torch.no_grad():
                # Compute each user_repr individually but minimize CUDA sync overhead
                user_reprs = torch.zeros(num_users, self.embedding_dim, device=self.device)
                for ui, items in enumerate(user_item_lists):
                    items_t = all_items_flat[offsets_t[ui]:offsets_t[ui + 1]]
                    user_reprs[ui] = self.model.item_emb(items_t).mean(dim=0)

            # --- On-the-fly triple generation + training ---
            # Instead of pre-building all triples, generate per batch on-the-fly
            bs = 4096  # larger batch for GPU efficiency
            n_batches = (total_triples + bs - 1) // bs

            # Pre-generate all positive item indices (shuffle for randomness)
            pos_indices = torch.randperm(total_triples, device=self.device)

            for batch_i in range(n_batches):
                b_start = batch_i * bs
                b_end = min(b_start + bs, total_triples)
                batch_ids = pos_indices[b_start:b_end]
                batch_size_actual = b_end - b_start

                # Map flat indices back to user and local item index
                # Binary search to find which user each index belongs to
                user_ids = torch.searchsorted(offsets_t, batch_ids, right=True) - 1
                local_ids = batch_ids - offsets_t[user_ids]

                # Gather user representations
                user_batch = user_reprs[user_ids]

                # Gather positive items
                pos_batch = all_items_flat[batch_ids]

                # Sample negative items (on GPU)
                neg_batch = torch.randint(1, self.num_items, (batch_size_actual,), device=self.device)

                # Rejection: for any negative that matches a positive, resample
                # (vectorized - very rare, <0.1% so one pass is enough)
                mask = (neg_batch == pos_batch)
                if mask.any():
                    n_replace = mask.sum().item()
                    neg_batch[mask] = torch.randint(1, self.num_items, (n_replace,), device=self.device)

                pos_score, neg_score = self.model(user_batch, pos_batch, neg_batch)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size_actual

            avg_loss = total_loss / total_triples
            logger.info(f"Epoch {epoch}: loss={avg_loss:.4f} ({total_triples} triples)")
            if monitor is not None:
                monitor.log(epoch=epoch, loss=avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        if not history:
            return []

        self.model.eval()
        with torch.no_grad():
            history_tensor = torch.tensor(history, dtype=torch.long, device=self.device)
            history_emb = self.model.item_emb(history_tensor)
            user_repr = history_emb.mean(dim=0, keepdim=True)

            all_items = torch.arange(1, self.num_items, device=self.device)
            all_emb = self.model.item_emb(all_items)
            scores = (user_repr * all_emb).sum(dim=-1)

            history_set = set(history)
            for i, item_id in enumerate(all_items.tolist()):
                if item_id in history_set:
                    scores[i] = -float("inf")

            top_indices = torch.topk(scores, top_k).indices
            return [all_items[i].item() for i in top_indices]

    def recommend_batch(self, histories: List[List[int]], top_k: int = 10) -> List[List[int]]:
        """Batch recommend for multiple users. Fully vectorized on GPU."""
        if not histories:
            return []

        self.model.eval()
        with torch.no_grad():
            all_items = torch.arange(1, self.num_items, device=self.device)
            all_emb = self.model.item_emb(all_items)  # [N, D]

            # Pad histories and compute user representations in batch
            max_len = max((len(h) for h in histories if h), default=0)
            if max_len == 0:
                return [[] for _ in histories]

            batch_size = len(histories)
            padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            valid_lens = []
            for i, h in enumerate(histories):
                if h:
                    padded[i, :len(h)] = torch.tensor(h, dtype=torch.long, device=self.device)
                    valid_lens.append(len(h))
                else:
                    valid_lens.append(0)

            emb = self.model.item_emb(padded)  # [B, L, D]
            mask = (padded > 0).float().unsqueeze(-1)  # [B, L, 1]
            user_reprs = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, D]

            # All scores at once: [B, D] @ [D, N] = [B, N]
            all_scores = user_reprs @ all_emb.T  # [B, N]

            # Mask history items per user (all_items is 1-indexed: position = item_id - 1)
            for i, h in enumerate(histories):
                for item_id in h:
                    pos = item_id - 1
                    if 0 <= pos < all_scores.size(1):
                        all_scores[i, pos] = -float("inf")

            # Top-k for all users
            topk = torch.topk(all_scores, top_k, dim=1)
            return [[all_items[idx].item() for idx in topk.indices[i]] for i in range(batch_size)]
