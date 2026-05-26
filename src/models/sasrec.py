import math
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import SeqRecDataset
from src.models.base import BaseRecommender
from src.utils.config import (
    BATCH_SIZE, DROPOUT, EMBEDDING_DIM, LEARNING_RATE,
    MAX_SEQ_LEN, NEG_SAMPLES, NUM_EPOCHS, NUM_HEADS, NUM_LAYERS, PATIENCE
)
from src.utils.logger import get_logger

logger = get_logger("sasrec")


class SASRecModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, max_seq_len: int,
                 num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=embedding_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        # bias=False to keep scoring as pure dot product, consistent with encode path
        self.out_linear = nn.Linear(embedding_dim, num_items, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, 0, 0.02)
        nn.init.normal_(self.pos_emb.weight, 0, 0.02)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape

        item_embs = self.item_emb(input_seq)
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        pos_embs = self.pos_emb(positions)

        x = self.dropout(item_embs + pos_embs)

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_seq.device)
        padding_mask = (input_seq == 0)

        x = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        logits = self.out_linear(x)
        return logits

    def encode(self, input_seq):
        """Encode sequence to hidden states, without output projection."""
        batch_size, seq_len = input_seq.shape
        item_embs = self.item_emb(input_seq)
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        pos_embs = self.pos_emb(positions)
        x = self.dropout(item_embs + pos_embs)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_seq.device)
        padding_mask = (input_seq == 0)
        x = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        return x

    def compute_scores(self, hidden, item_ids):
        """Compute scores for specific items using the output weight matrix."""
        weight = self.out_linear.weight  # [num_items, D]
        item_vecs = weight[item_ids]  # [batch, D]
        return (hidden * item_vecs).sum(dim=-1)


class SASRecRecommender(BaseRecommender):
    def __init__(self, num_items: int, embedding_dim: int = EMBEDDING_DIM,
                 max_seq_len: int = MAX_SEQ_LEN, num_heads: int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS, dropout: float = 0.5,
                 lr: float = 1e-3, num_epochs: int = NUM_EPOCHS,
                 batch_size: int = 2048, neg_samples: int = NEG_SAMPLES,
                 patience: int = 5, weight_decay: float = 1e-6):
        super().__init__(num_items)
        self.name = "SASRec"
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.patience = patience
        self.weight_decay = weight_decay
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, monitor=None, **kwargs):
        category = kwargs.get("category", "default")
        valid_data = kwargs.get("valid_data", None)
        checkpoint_dir = "results"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{category}_SASRec_checkpoint.pt")

        self.model = SASRecModel(
            self.num_items, self.embedding_dim, self.max_seq_len,
            self.num_heads, self.num_layers, self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # We pass neg_samples=1 to dataset to minimize CPU-side sampling overhead
        dataset = SeqRecDataset(train_data, self.max_seq_len, self.num_items, neg_samples=1)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=0, pin_memory=True)

        best_val = -float("inf")
        best_state = None
        patience_counter = 0
        start_epoch = 0
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        # Skip checkpoint resume — val-based early stopping must run from a clean state to track best epoch.

        for epoch in range(start_epoch, self.num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                input_seq = batch["input_seq"].to(self.device)            # [B, L]
                target = batch["target"].to(self.device)                  # [B]
                seq_lens = batch["seq_len"].to(self.device)               # [B]

                B, L = input_seq.shape

                # --- Full-sequence next-item labels (right-padding) ---
                # labels[b, i] = next item after position i:
                #   - input_seq[b, i+1] for i < seq_lens[b] - 1
                #   - target[b]         for i == seq_lens[b] - 1
                #   - 0 (ignored)       for i >= seq_lens[b]
                labels = torch.zeros_like(input_seq)
                labels[:, :-1] = input_seq[:, 1:]  # shift-left
                batch_idx = torch.arange(B, device=self.device)
                labels[batch_idx, seq_lens - 1] = target
                loss_mask = (labels != 0)  # [B, L] valid prediction positions

                # One negative per (b, i); cheap rejection against same-position positive.
                negatives = torch.randint(1, self.num_items, (B, L), device=self.device)
                conflict = (negatives == labels) & loss_mask
                if conflict.any():
                    n_replace = int(conflict.sum().item())
                    negatives[conflict] = torch.randint(1, self.num_items, (n_replace,), device=self.device)

                hidden = self.model.encode(input_seq)  # [B, L, D]

                out_w = self.model.out_linear.weight   # [num_items, D]
                pos_emb = out_w[labels]                # [B, L, D]
                neg_emb = out_w[negatives]             # [B, L, D]
                pos_scores = (hidden * pos_emb).sum(dim=-1)  # [B, L]
                neg_scores = (hidden * neg_emb).sum(dim=-1)  # [B, L]

                loss_per_pos = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)  # [B, L]
                mask_f = loss_mask.float()
                loss = (loss_per_pos * mask_f).sum() / mask_f.sum().clamp(min=1)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches % 100 == 0:
                    logger.info(f"  Epoch {epoch} batch {num_batches}/{total_batches} loss={loss.item():.4f}")

            avg_loss = total_loss / max(num_batches, 1)

            # Validation-based early stopping
            val_ndcg = None
            if valid_data is not None:
                from src.evaluation.evaluator import Evaluator
                from src.utils.config import TOP_K
                self.model.eval()
                val_metrics = Evaluator(self, valid_data, self.num_items, TOP_K).evaluate()
                val_ndcg = val_metrics[f"NDCG@{TOP_K}"]

            if monitor:
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
            seq = history[-self.max_seq_len:]
            seq_len = len(seq)
            # Right padding to match training
            padded = seq + [0] * (self.max_seq_len - seq_len)
            input_tensor = torch.tensor([padded], dtype=torch.long, device=self.device)

            # Use encode() + out_linear for efficiency, same as training
            hidden = self.model.encode(input_tensor)          # [1, L, D]
            last_hidden = hidden[0, seq_len - 1, :]            # [D]
            scores = self.model.out_linear(last_hidden)        # [num_items]

            for item_id in history:
                if 0 < item_id < self.num_items:
                    scores[item_id] = -float("inf")
            scores[0] = -float("inf")

            top_indices = torch.topk(scores, top_k).indices
            return top_indices.tolist()

    def recommend_batch(self, histories: List[List[int]], top_k: int = 10) -> List[List[int]]:
        """Batch recommend for multiple users. Fully vectorized on GPU."""
        if not histories:
            return []

        self.model.eval()
        with torch.no_grad():
            batch_size = len(histories)

            # Pad all histories to max_seq_len (right-padded)
            padded_seqs = []
            seq_lens = []
            for h in histories:
                seq = h[-self.max_seq_len:] if h else []
                seq_len = len(seq)
                padded = seq + [0] * (self.max_seq_len - seq_len)
                padded_seqs.append(padded)
                seq_lens.append(max(seq_len, 1)) # avoid 0 length

            input_tensor = torch.tensor(padded_seqs, dtype=torch.long, device=self.device) # [B, L]
            seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=self.device) # [B]

            # Process in chunks of 512 to avoid any potential GPU memory issue
            chunk_size = 512
            all_scores = []

            for chunk_idx in range(0, batch_size, chunk_size):
                chunk_input = input_tensor[chunk_idx : chunk_idx + chunk_size]
                chunk_lens = seq_lens_t[chunk_idx : chunk_idx + chunk_size]

                # Get hidden states [chunk_B, L, D]
                hidden = self.model.encode(chunk_input)
                # Extract last position for each sequence in right-padding: chunk_lens - 1
                last_hidden = hidden[torch.arange(hidden.size(0), device=self.device), chunk_lens - 1, :] # [chunk_B, D]
                # Project
                chunk_scores = self.model.out_linear(last_hidden) # [chunk_B, num_items]
                all_scores.append(chunk_scores)

            all_scores = torch.cat(all_scores, dim=0) # [B, num_items]

            # Mask history items per user
            for i, h in enumerate(histories):
                if h:
                    for item_id in h:
                        if 0 < item_id < self.num_items:
                            all_scores[i, item_id] = -float("inf")
                all_scores[i, 0] = -float("inf") # Mask padding item

            top_indices = torch.topk(all_scores, top_k, dim=1).indices
            return top_indices.tolist()
