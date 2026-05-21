with open("/root/recsys/src/models/sasrec.py", "r") as f:
    code = f.read()

# Remove weight tying
code = code.replace(
    '        self.out_linear = nn.Linear(embedding_dim, num_items)\n        # Weight tying: output projection shares weights with item embeddings\n        self.out_linear.weight = self.item_emb.weight',
    '        self.out_linear = nn.Linear(embedding_dim, num_items)'
)

# Remove NaN check, restore original
code = code.replace(
    '''                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Check for NaN
                valid = True
                for p in self.model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        valid = False
                        break
                if valid:
                    optimizer.step()
                else:
                    optimizer.zero_grad()''',
    '''                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()'''
)

# Update mask
code = code.replace(
    '        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_seq.device)\n        padding_mask = (input_seq == 0)',
    '        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_seq.device), diagonal=1)\n        padding_mask = (input_seq == 0)'
)

with open("/root/recsys/src/models/sasrec.py", "w") as f:
    f.write(code)
print("Fixed!")
