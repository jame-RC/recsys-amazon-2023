import re

with open("/root/recsys/src/models/sasrec.py", "r") as f:
    code = f.read()

# Replace the encode/compute_scores training block with original forward approach
old_block = """                # Efficient: encode without output projection, then score only needed items
                hidden = self.model.encode(input_seq)  # [B, L, D]
                last_hidden = hidden[:, -1, :]  # [B, D]

                # Score positive and negative items
                all_targets = torch.cat([target.unsqueeze(1), negatives], dim=1)  # [B, 1+N]
                scores = self.model.compute_scores(last_hidden.unsqueeze(1), all_targets)  # [B, 1+N]
                pos_scores = scores[:, 0]
                neg_scores = scores[:, 1:]"""

new_block = """                logits = self.model(input_seq)
                last_logits = logits[:, -1, :]

                pos_scores = last_logits.gather(1, target.unsqueeze(1))
                neg_scores = last_logits.gather(1, negatives)"""

code = code.replace(old_block, new_block)

with open("/root/recsys/src/models/sasrec.py", "w") as f:
    f.write(code)
print("SASRec fix applied!")
