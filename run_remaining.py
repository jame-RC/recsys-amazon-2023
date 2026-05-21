import sys, json, os, time, torch
sys.path.insert(0, '/root/recsys')

from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.models.pop import PopRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.bpr import BPRRecommender
from src.utils.config import TOP_K

CATEGORIES = ['Musical_Instruments', 'CDs_and_Vinyl']
RESULTS_DIR = '/root/recsys/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {}

# Load existing results to continue
for f in os.listdir(RESULTS_DIR):
    if f == 'all_results.json':
        results.update(json.load(open(f'{RESULTS_DIR}/{f}')))
    elif f.endswith('_model.pt'):
        pass  # Skip model files

for cat in CATEGORIES:
    print(f'\n{"="*60}', flush=True)
    print(f'  CATEGORY: {cat}', flush=True)
    print(f'{"="*60}', flush=True)
    
    dataset = AmazonDataset(cat)
    train_data = dataset.get_train_sequences()
    valid_data = dataset.get_eval_data('valid')
    test_data = dataset.get_eval_data('test')
    num_items = len(dataset.item_vocab)
    print(f'  Items: {num_items}', flush=True)
    print(f'  Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}', flush=True)

    # --- Popularity ---
    print(f'\n  >>> Popularity', flush=True)
    m = PopRecommender(num_items)
    t0 = time.time()
    m.fit(train_data)
    print(f'    Train: {time.time()-t0:.1f}s', flush=True)
    vm = Evaluator(m, valid_data, num_items, TOP_K).evaluate()
    tm = Evaluator(m, test_data, num_items, TOP_K).evaluate()
    results[f'{cat}_Popularity'] = {'category': cat, 'model': 'Popularity', 'valid': vm, 'test': tm}
    torch.save(m, f'{RESULTS_DIR}/{cat}_Popularity_model.pt')
    print(f'    Saved', flush=True)

    # --- ItemCF ---
    print(f'\n  >>> ItemCF', flush=True)
    m = ItemCFRecommender(num_items)
    t0 = time.time()
    m.fit(train_data)
    print(f'    Train: {time.time()-t0:.1f}s', flush=True)
    vm = Evaluator(m, valid_data, num_items, TOP_K).evaluate()
    if cat != 'CDs_and_Vinyl':
        tm = Evaluator(m, test_data, num_items, TOP_K).evaluate()
    else:
        tm = {'NDCG@10': -1, 'Hit@10': -1, 'MRR@10': -1, 'eval_time_sec': -1}
        print(f'    Skipping test eval (too slow for CDs)', flush=True)
    results[f'{cat}_ItemCF'] = {'category': cat, 'model': 'ItemCF', 'valid': vm, 'test': tm}
    torch.save(m, f'{RESULTS_DIR}/{cat}_ItemCF_model.pt')
    print(f'    Saved', flush=True)

    # --- BPR ---
    print(f'\n  >>> BPR', flush=True)
    m = BPRRecommender(num_items, num_epochs=30)
    t0 = time.time()
    m.fit(train_data)
    print(f'    Train: {time.time()-t0:.1f}s', flush=True)
    vm = Evaluator(m, valid_data, num_items, TOP_K).evaluate()
    tm = Evaluator(m, test_data, num_items, TOP_K).evaluate()
    results[f'{cat}_BPR'] = {'category': cat, 'model': 'BPR', 'valid': vm, 'test': tm}
    torch.save(m.model.state_dict(), f'{RESULTS_DIR}/{cat}_BPR_model.pt')
    print(f'    Saved', flush=True)

# Save all results
with open(f'{RESULTS_DIR}/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{"="*60}', flush=True)
print(f'  FINAL', flush=True)
print(f'{"="*60}', flush=True)
for key in sorted(results.keys()):
    d = results[key]
    print(f'  {d["model"]:12s} | {d["category"]:25s} | NDCG@10={d["test"]["NDCG@10"]:.4f}', flush=True)
print(f'\nDone!', flush=True)
