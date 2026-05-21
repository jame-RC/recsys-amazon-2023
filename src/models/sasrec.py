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
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT,
                 lr: float = 1e-4, num_epochs: int = NUM_EPOCHS,
                 batch_size: int = 2048, neg_samples: int = NEG_SAMPLES,
                 patience: int = PATIENCE):
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
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, monitor=None, **kwargs):
        category = kwargs.get("category", "default")
        checkpoint_dir = "results"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{category}_SASRec_checkpoint.pt")

        self.model = SASRecModel(
            self.num_items, self.embedding_dim, self.max_seq_len,
            self.num_heads, self.num_layers, self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # We pass neg_samples=1 to dataset to minimize CPU-side sampling overhead
        dataset = SeqRecDataset(train_data, self.max_seq_len, self.num_items, neg_samples=1)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=0, pin_memory=True)

        best_loss = float("inf")
        patience_counter = 0
        start_epoch = 0
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        # Try to resume from checkpoint
        if os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint['best_loss']
                patience_counter = checkpoint['patience_counter']
                logger.info(f"Successfully resumed training from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting training from scratch.")

        for epoch in range(start_epoch, self.num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                input_seq = batch["input_seq"].to(self.device)
                target = batch["target"].to(self.device)
                seq_lens = batch["seq_len"]

                # Vectorized negative sampling on GPU - completely eliminates CPU bottleneck
                batch_size_actual = input_seq.size(0)
                negatives = torch.randint(1, self.num_items, (batch_size_actual, self.neg_samples), device=self.device)

                # BPR loss: score positive items higher than negatives
                hidden = self.model.encode(input_seq)            # [B, L, D]
                # Use last NON-PADDING position (Padding is on the right)
                last_hidden = hidden[torch.arange(hidden.size(0), device=self.device), seq_lens - 1, :]  # [B, D]

                pos_scores = self.model.compute_scores(
                    last_hidden.unsqueeze(1), target.unsqueeze(1)
                ).squeeze(1)
                neg_scores = self.model.compute_scores(
                    last_hidden.unsqueeze(1), negatives
                ).mean(dim=1)

                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches % 100 == 0:
                    logger.info(f"  Epoch {epoch} batch {num_batches}/{total_batches} loss={loss.item():.4f}")

            avg_loss = total_loss / max(num_batches, 1)
            if monitor:
                monitor.log(epoch=epoch, loss=avg_loss)
            logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Save checkpoint at the end of each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, checkpoint_path)

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Remove temporary checkpoint on successful completion
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logger.info("Training completed. Temporary checkpoint removed.")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint file: {e}")

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
