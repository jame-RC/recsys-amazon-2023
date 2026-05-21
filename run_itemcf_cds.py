import sys, json, os, time, math
from collections import defaultdict

sys.path.insert(0, '/root/recsys')

from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.models.item_cf import ItemCFRecommender
from src.utils.config import TOP_K

cat = 'CDs_and_Vinyl'
dataset = AmazonDataset(cat)
train_data = dataset.get_train_sequences()
valid_data = dataset.get_eval_data('valid')
test_data = dataset.get_eval_data('test')
num_items = len(dataset.item_vocab)
print(f'Items: {num_items}, Train: {len(train_data)}, Valid: {len(valid_data)}', flush=True)

print('ItemCF training...', flush=True)
m = ItemCFRecommender(num_items)
t0 = time.time()
m.fit(train_data)
print(f'Train: {time.time()-t0:.1f}s', flush=True)

print('Validation eval...', flush=True)
t0 = time.time()
vm = Evaluator(m, valid_data[:5000], num_items, TOP_K).evaluate()
print(f'Valid(5K): NDCG={vm["NDCG@10"]:.4f}, time={time.time()-t0:.1f}s', flush=True)

print('Saving model...', flush=True)
import torch
torch.save(m, f'/root/recsys/results/{cat}_ItemCF_model.pt')
print('Done!', flush=True)
