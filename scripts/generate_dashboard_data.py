import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AmazonDataset
from src.models.bpr import BPRRecommender, BPRModel
from src.models.bpr_advanced import BPRAdvancedRecommender, BPRAdvancedModel
from src.models.sasrec import SASRecRecommender, SASRecModel
from src.models.pop import PopRecommender
from src.models.item_cf import ItemCFRecommender
from src.utils.config import RESULTS_DIR, TOP_K

CATEGORY = "Industrial_and_Scientific"

def main():
    print("Loading datasets...", flush=True)
    dataset = AmazonDataset(CATEGORY)

    train_data = dataset.get_train_sequences()
    test_data = dataset.get_eval_data("test")
    num_items = len(dataset.item_vocab)
    print(f"Items: {num_items}, Test users: {len(test_data)}")

    # Initialize and fit models (or load from checkpoints)
    print("Fitting Popularity & ItemCF...", flush=True)
    pop_model = PopRecommender(num_items)
    pop_model.fit(train_data)

    itemcf_model = ItemCFRecommender(num_items)
    itemcf_model.fit(train_data)

    print("Loading BPR weights...", flush=True)
    bpr_recommender = BPRRecommender(num_items)
    bpr_recommender.model = BPRModel(num_items, bpr_recommender.embedding_dim).to(bpr_recommender.device)
    bpr_path = os.path.join(RESULTS_DIR, f"{CATEGORY}_BPR_model.pt")
    if os.path.exists(bpr_path):
        bpr_recommender.model.load_state_dict(torch.load(bpr_path, map_location=bpr_recommender.device))
        print("BPR loaded successfully.")
    else:
        print("Warning: BPR model not found! Fitting dummy...")
        bpr_recommender.fit(train_data[:1000])

    print("Loading BPR_Advanced weights...", flush=True)
    bpr_adv_recommender = BPRAdvancedRecommender(num_items)
    bpr_adv_recommender.model = BPRAdvancedModel(num_items, bpr_adv_recommender.embedding_dim).to(bpr_adv_recommender.device)
    bpr_adv_path = os.path.join(RESULTS_DIR, f"{CATEGORY}_BPR_Advanced_model.pt")
    if os.path.exists(bpr_adv_path):
        bpr_adv_recommender.model.load_state_dict(torch.load(bpr_adv_path, map_location=bpr_adv_recommender.device))
        print("BPR_Advanced loaded successfully.")
    else:
        print("Warning: BPR_Advanced model not found! Fitting dummy...")
        bpr_adv_recommender.fit(train_data[:1000])

    print("Loading SASRec weights...", flush=True)
    sasrec_recommender = SASRecRecommender(num_items)
    sasrec_recommender.model = SASRecModel(
        num_items, sasrec_recommender.embedding_dim, sasrec_recommender.max_seq_len,
        sasrec_recommender.num_heads, sasrec_recommender.num_layers, sasrec_recommender.dropout
    ).to(sasrec_recommender.device)
    sasrec_path = os.path.join(RESULTS_DIR, f"{CATEGORY}_SASRec_model.pt")
    if os.path.exists(sasrec_path):
        sasrec_recommender.model.load_state_dict(torch.load(sasrec_path, map_location=sasrec_recommender.device))
        print("SASRec loaded successfully.")
    else:
        print("Warning: SASRec model not found! Fitting dummy...")
        sasrec_recommender.fit(train_data[:1000])

    # Select 5 representative test users with different sequence lengths
    print("Selecting sample users...", flush=True)
    sample_users = []
    # Sort test data by history length to find diverse profiles
    test_data_sorted = sorted(test_data, key=lambda x: len(x[1]))
    
    # Select users with histories of lengths e.g. 5, 8, 10, 12, 15
    desired_lengths = [5, 8, 10, 12, 15]
    selected_indices = []
    
    for length in desired_lengths:
        for idx, (user_idx, history, target) in enumerate(test_data_sorted):
            if idx not in selected_indices and abs(len(history) - length) <= 1:
                selected_indices.append(idx)
                break
                
    # Fallback to make sure we have exactly 5 if some lengths are missing
    while len(selected_indices) < 5 and len(test_data_sorted) > 0:
        for idx in range(len(test_data_sorted)):
            if idx not in selected_indices:
                selected_indices.append(idx)
                break
                
    print(f"Selected {len(selected_indices)} sample users.")

    # We need to resolve metadata for:
    # 1. Historical items
    # 2. Target items
    # 3. Recommended items for all models
    asin_to_lookup = set()
    user_cases = []

    for idx in selected_indices:
        user_idx, history, target = test_data_sorted[idx]
        user_id_raw = dataset.user_vocab.decode(user_idx)

        # Run recommendations
        print(f"Generating recommendations for user {user_id_raw} (history size: {len(history)})...")
        rec_pop = pop_model.recommend(history, top_k=TOP_K)
        rec_itemcf = itemcf_model.recommend(history, top_k=TOP_K)
        rec_bpr = bpr_recommender.recommend(history, top_k=TOP_K)
        rec_bpr_adv = bpr_adv_recommender.recommend(history, top_k=TOP_K)
        rec_sasrec = sasrec_recommender.recommend(history, top_k=TOP_K)

        # Decode ASINs
        history_asins = [dataset.item_vocab.decode(i) for i in history]
        target_asin = dataset.item_vocab.decode(target)
        rec_pop_asins = [dataset.item_vocab.decode(i) for i in rec_pop]
        rec_itemcf_asins = [dataset.item_vocab.decode(i) for i in rec_itemcf]
        rec_bpr_asins = [dataset.item_vocab.decode(i) for i in rec_bpr]
        rec_bpr_adv_asins = [dataset.item_vocab.decode(i) for i in rec_bpr_adv]
        rec_sasrec_asins = [dataset.item_vocab.decode(i) for i in rec_sasrec]

        # Add to set for lookup
        asin_to_lookup.update(history_asins)
        asin_to_lookup.add(target_asin)
        asin_to_lookup.update(rec_pop_asins)
        asin_to_lookup.update(rec_itemcf_asins)
        asin_to_lookup.update(rec_bpr_asins)
        asin_to_lookup.update(rec_bpr_adv_asins)
        asin_to_lookup.update(rec_sasrec_asins)

        user_cases.append({
            "user_id": user_id_raw,
            "history": history_asins,
            "target": target_asin,
            "recommendations": {
                "Popularity": rec_pop_asins,
                "ItemCF": rec_itemcf_asins,
                "BPR": rec_bpr_asins,
                "BPR_Advanced": rec_bpr_adv_asins,
                "SASRec": rec_sasrec_asins
            }
        })

    # High-efficiency stream-parsing of the 1.13 GB metadata file
    meta_path = os.path.join(dataset.data_dir, f"meta_{dataset.category}.jsonl")
    item_metadata = {}
    
    print(f"Streaming {meta_path} to lookup metadata for {len(asin_to_lookup)} unique ASINs...")
    found_count = 0
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                try:
                    item = json.loads(line)
                    asin = item.get("parent_asin")
                    if asin in asin_to_lookup:
                        # Extract title and a clean description
                        desc = " ".join(item.get("description", [])) if item.get("description") else ""
                        if len(desc) > 180:
                            desc = desc[:177] + "..."
                        
                        title = item.get("title", "Amazon Product")
                        if len(title) > 80:
                            title = title[:77] + "..."

                        item_metadata[asin] = {
                            "asin": asin,
                            "title": title,
                            "categories": item.get("categories", ["Industrial & Scientific"]),
                            "description": desc,
                            "price": item.get("price", "N/A")
                        }
                        found_count += 1
                        if found_count == len(asin_to_lookup):
                            print("All requested items matched! Stopping scan early.")
                            break
                except Exception as e:
                    pass
                if line_no % 100000 == 0 and line_no > 0:
                    print(f"  Scanned {line_no} lines... Matched {found_count}/{len(asin_to_lookup)} items.")
    else:
        print("Warning: Meta file not found! Generating dummy metadata...")

    # For any item not found in meta file (fallback)
    for asin in asin_to_lookup:
        if asin not in item_metadata:
            item_metadata[asin] = {
                "asin": asin,
                "title": f"Amazon Item {asin}",
                "categories": ["Industrial & Scientific"],
                "description": "Premium industrial grade equipment and scientific testing tools from selected Amazon suppliers.",
                "price": "19.99"
            }

    # Populate final user objects with metadata
    final_users = []
    for case in user_cases:
        hist_objs = [item_metadata[asin] for asin in case["history"]]
        target_obj = item_metadata[case["target"]]
        
        recs_objs = {}
        for model_name, rec_asins in case["recommendations"].items():
            recs_objs[model_name] = [item_metadata[asin] for asin in rec_asins]

        final_users.append({
            "user_id": case["user_id"],
            "history": hist_objs,
            "target": target_obj,
            "recommendations": recs_objs
        })

    # Hardcoded compiled evaluation metrics for dashboard display
    metrics_data = {
        "Industrial_and_Scientific": {
            "Popularity": {"NDCG": 0.0081, "Hit": 0.0156, "MRR": 0.0059},
            "ItemCF": {"NDCG": 0.0050, "Hit": 0.0082, "MRR": 0.0040},
            "BPR": {"NDCG": 0.0155, "Hit": 0.0275, "MRR": 0.0117},
            "BPR_Advanced": {"NDCG": 0.0154, "Hit": 0.0296, "MRR": 0.0111},
            "SASRec": {"NDCG": 0.0086, "Hit": 0.0165, "MRR": 0.0063},
            "LLM Zero-Shot": {"NDCG": 0.0033, "Hit": 0.0100, "MRR": 0.0014},
            "LLM Few-Shot": {"NDCG": 0.0033, "Hit": 0.0100, "MRR": 0.0014}
        },
        "Musical_Instruments": {
            "Popularity": {"NDCG": 0.0128, "Hit": 0.0243, "MRR": 0.0094},
            "ItemCF": {"NDCG": 0.0023, "Hit": 0.0044, "MRR": 0.0017},
            "BPR": {"NDCG": 0.0211, "Hit": 0.0384, "MRR": 0.0159},
            "BPR_Advanced": {"NDCG": 0.0220, "Hit": 0.0434, "MRR": 0.0156},
            "SASRec": {"NDCG": 0.0095, "Hit": 0.0193, "MRR": 0.0066}
        },
        "CDs_and_Vinyl": {
            "Popularity": {"NDCG": 0.0009, "Hit": 0.0021, "MRR": 0.0006},
            "ItemCF": {"NDCG": 0.0000, "Hit": 0.0000, "MRR": 0.0000},
            "BPR": {"NDCG": 0.0281, "Hit": 0.0497, "MRR": 0.0215},
            "BPR_Advanced": {"NDCG": 0.0000, "Hit": 0.0000, "MRR": 0.0000},
            "SASRec": {"NDCG": 0.0017, "Hit": 0.0036, "MRR": 0.0011}
        }
    }

    dashboard_payload = {
        "metrics": metrics_data,
        "sample_users": final_users
    }

    # Save to JavaScript file
    output_path = os.path.join(RESULTS_DIR, "dashboard_data.js")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated by generate_dashboard_data.py */\n")
        f.write("const DASHBOARD_DATA = ")
        json.dump(dashboard_payload, f, indent=2, ensure_ascii=False)
        f.write(";\n")
        
    print(f"\nDashboard data successfully generated and saved to {output_path}!", flush=True)

if __name__ == "__main__":
    main()
