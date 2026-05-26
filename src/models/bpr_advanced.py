from typing import List
import torch
import torch.nn as nn
import numpy as np

from src.models.base import BaseRecommender
from src.utils.config import EMBEDDING_DIM, LEARNING_RATE, NUM_EPOCHS, PATIENCE
from src.utils.logger import get_logger

logger = get_logger("bpr_advanced")


class BPRAdvancedModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)
        # Learnable parameter to adaptively balance short-term interest (last item) and long-term interest (average)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_repr, pos_items, neg_items):
        pos_emb = self.item_emb(pos_items)
        neg_emb = self.item_emb(neg_items)
        
        pos_bias = self.item_bias(pos_items).squeeze(-1)
        neg_bias = self.item_bias(neg_items).squeeze(-1)
        
        pos_score = (user_repr * pos_emb).sum(dim=-1) + pos_bias
        neg_score = (user_repr * neg_emb).sum(dim=-1) + neg_bias
        return pos_score, neg_score


class BPRAdvancedRecommender(BaseRecommender):
    def __init__(self, num_items: int, embedding_dim: int = EMBEDDING_DIM, lr: float = 1e-3,
                 num_epochs: int = 50, patience: int = 5, weight_decay: float = 1e-6,
                 neg_samples: int = 5):
        super().__init__(num_items)
        self.name = "BPR_Advanced"
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.neg_samples = neg_samples
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, monitor=None, **kwargs):
        import os
        category = kwargs.get("category", "default")
        valid_data = kwargs.get("valid_data", None)
        self.model = BPRAdvancedModel(self.num_items, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Build tensor dataset on GPU for ultimate speed
        # Each entry in train_data: (user_idx, history, target)
        histories = []
        targets = []
        
        max_len = 0
        for _, history, target in train_data:
            if history:
                # Cap history to the last 100 items to prevent huge memory footprint
                truncated_history = history[-100:]
                histories.append(truncated_history)
                targets.append(target)
                max_len = max(max_len, len(truncated_history))
        
        if not histories:
            return
            
        num_samples = len(histories)
        
        # Build padded history and target tensors
        padded_histories = np.zeros((num_samples, max_len), dtype=np.int64)
        history_lens = np.zeros(num_samples, dtype=np.int64)
        for i, h in enumerate(histories):
            padded_histories[i, :len(h)] = h
            history_lens[i] = len(h)
            
        padded_histories_t = torch.tensor(padded_histories, dtype=torch.long, device=self.device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=self.device)
        history_lens_t = torch.tensor(history_lens, dtype=torch.long, device=self.device)
        
        best_val = -float("inf")
        best_state = None
        patience_counter = 0
        K = self.neg_samples

        logger.info(f"Start training BPR_Advanced with {num_samples} samples, neg_samples={K}...")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            
            # Shuffle batch indices
            indices = torch.randperm(num_samples, device=self.device)
            
            bs = 2048
            n_batches = (num_samples + bs - 1) // bs
            
            for batch_i in range(n_batches):
                b_start = batch_i * bs
                b_end = min(b_start + bs, num_samples)
                batch_indices = indices[b_start:b_end]
                batch_size_actual = b_end - b_start
                
                # Gather inputs
                b_history = padded_histories_t[batch_indices] # [B, L]
                b_target = targets_t[batch_indices] # [B]
                b_lens = history_lens_t[batch_indices] # [B]
                
                # Compute long-term interest (average embedding of history items)
                emb = self.model.item_emb(b_history) # [B, L, D]
                mask = (b_history > 0).float().unsqueeze(-1) # [B, L, 1]
                long_term_emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # [B, D]
                
                # Compute short-term interest (the last item embedding in history)
                last_item_idx = b_lens - 1
                short_term_emb = emb[torch.arange(batch_size_actual, device=self.device), last_item_idx] # [B, D]
                
                # Dynamic User representation fusion using adaptive alpha
                alpha = torch.sigmoid(self.model.alpha) # clamp alpha in [0, 1]
                user_repr = alpha * short_term_emb + (1 - alpha) * long_term_emb # [B, D]
                
                # Sample K negative items per positive: [B, K]
                neg_batch = torch.randint(1, self.num_items, (batch_size_actual, K), device=self.device)
                # Cheap rejection against the positive
                mask_conflict = (neg_batch == b_target.unsqueeze(1))
                if mask_conflict.any():
                    n_replace = int(mask_conflict.sum().item())
                    neg_batch[mask_conflict] = torch.randint(1, self.num_items, (n_replace,), device=self.device)

                # Pos score: [B]
                pos_emb = self.model.item_emb(b_target)
                pos_bias = self.model.item_bias(b_target).squeeze(-1)
                pos_score = (user_repr * pos_emb).sum(dim=-1) + pos_bias

                # Neg scores: [B, K] — broadcast user_repr against K negatives
                neg_emb = self.model.item_emb(neg_batch)              # [B, K, D]
                neg_bias = self.model.item_bias(neg_batch).squeeze(-1)  # [B, K]
                neg_score = (user_repr.unsqueeze(1) * neg_emb).sum(dim=-1) + neg_bias  # [B, K]

                # Pairwise BPR loss averaged over K negatives, then over batch
                loss = -torch.log(torch.sigmoid(pos_score.unsqueeze(1) - neg_score) + 1e-10).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_size_actual
                
            avg_loss = total_loss / num_samples

            # Validation-based early stopping
            val_ndcg = None
            if valid_data is not None:
                from src.evaluation.evaluator import Evaluator
                from src.utils.config import TOP_K
                self.model.eval()
                val_metrics = Evaluator(self, valid_data, self.num_items, TOP_K).evaluate()
                val_ndcg = val_metrics[f"NDCG@{TOP_K}"]

            if monitor is not None:
                if val_ndcg is not None:
                    monitor.log(epoch=epoch, loss=avg_loss, val_ndcg=val_ndcg)
                else:
                    monitor.log(epoch=epoch, loss=avg_loss)
            if val_ndcg is not None:
                logger.info(f"Epoch {epoch}: loss={avg_loss:.4f} val_NDCG@10={val_ndcg:.4f}")
            else:
                logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}")

            if val_ndcg is not None:
                if val_ndcg > best_val:
                    best_val = val_ndcg
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}, best val NDCG@10={best_val:.4f}")
                    break

        # Restore best validation checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"Restored best checkpoint with val NDCG@10={best_val:.4f}")

    def recommend(self, history: List[int], top_k: int = 10) -> List[int]:
        if not history:
            return []

        self.model.eval()
        with torch.no_grad():
            history_tensor = torch.tensor(history, dtype=torch.long, device=self.device)
            history_emb = self.model.item_emb(history_tensor)
            
            # Compute long-term (average) and short-term (last item)
            long_term_emb = history_emb.mean(dim=0, keepdim=True)
            short_term_emb = history_emb[-1].unsqueeze(0)
            
            alpha = torch.sigmoid(self.model.alpha)
            user_repr = alpha * short_term_emb + (1 - alpha) * long_term_emb # [1, D]

            all_items = torch.arange(1, self.num_items, device=self.device)
            all_emb = self.model.item_emb(all_items) # [N, D]
            all_bias = self.model.item_bias(all_items).squeeze(-1) # [N]
            
            scores = (user_repr * all_emb).sum(dim=-1) + all_bias # [N]

            history_set = set(history)
            for i, item_id in enumerate(all_items.tolist()):
                if item_id in history_set:
                    scores[i] = -float("inf")

            top_indices = torch.topk(scores, top_k).indices
            return [all_items[i].item() for i in top_indices]

    def recommend_batch(self, histories: List[List[int]], top_k: int = 10) -> List[List[int]]:
        """Batch recommend using dynamic long-term & short-term user representation. Fully vectorized on GPU."""
        if not histories:
            return []

        self.model.eval()
        with torch.no_grad():
            all_items = torch.arange(1, self.num_items, device=self.device)
            all_emb = self.model.item_emb(all_items)  # [N, D]
            all_bias = self.model.item_bias(all_items).squeeze(-1)  # [N]

            # Truncate histories to the last 100 items to avoid excessive GPU memory allocation
            truncated_histories = [h[-100:] if h else [] for h in histories]
            max_len = max((len(h) for h in truncated_histories if h), default=0)
            if max_len == 0:
                return [[] for _ in histories]

            batch_size = len(histories)
            padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            valid_lens = []
            for i, h in enumerate(truncated_histories):
                if h:
                    padded[i, :len(h)] = torch.tensor(h, dtype=torch.long, device=self.device)
                    valid_lens.append(len(h))
                else:
                    valid_lens.append(0)

            # [B, L, D]
            emb = self.model.item_emb(padded)
            
            # Compute long term
            mask = (padded > 0).float().unsqueeze(-1)  # [B, L, 1]
            long_term_emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, D]
            
            # Compute short term
            last_indices = torch.tensor([max(0, l - 1) for l in valid_lens], dtype=torch.long, device=self.device)
            short_term_emb = emb[torch.arange(batch_size, device=self.device), last_indices]  # [B, D]
            
            # Fusion
            alpha = torch.sigmoid(self.model.alpha)
            user_reprs = alpha * short_term_emb + (1 - alpha) * long_term_emb # [B, D]

            # [B, N]
            all_scores = user_reprs @ all_emb.T + all_bias.unsqueeze(0)

            # Mask histories
            for i, h in enumerate(histories):
                for item_id in h:
                    pos = item_id - 1
                    if 0 <= pos < all_scores.size(1):
                        all_scores[i, pos] = -float("inf")

            topk = torch.topk(all_scores, top_k, dim=1)
            return [[all_items[idx].item() for idx in topk.indices[i]] for i in range(batch_size)]
