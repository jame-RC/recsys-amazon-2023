import json, os

results = {
    "Industrial_and_Scientific_Popularity": {
        "category": "Industrial_and_Scientific", "model": "Popularity",
        "test": {"NDCG@10": 0.0081, "Hit@10": 0.0156, "MRR@10": 0.0059},
        "valid": {"NDCG@10": 0.0093, "Hit@10": 0.0171, "MRR@10": 0.0071}
    },
    "Industrial_and_Scientific_ItemCF": {
        "category": "Industrial_and_Scientific", "model": "ItemCF",
        "test": {"NDCG@10": 0.0050, "Hit@10": 0.0082, "MRR@10": 0.0040},
        "valid": {"NDCG@10": 0.0068, "Hit@10": 0.0112, "MRR@10": 0.0055}
    },
    "Industrial_and_Scientific_BPR": {
        "category": "Industrial_and_Scientific", "model": "BPR",
        "test": {"NDCG@10": 0.0155, "Hit@10": 0.0275, "MRR@10": 0.0117},
        "valid": {"NDCG@10": 0.0210, "Hit@10": 0.0374, "MRR@10": 0.0159}
    },
    "Musical_Instruments_Popularity": {
        "category": "Musical_Instruments", "model": "Popularity",
        "test": {"NDCG@10": 0.0128, "Hit@10": 0.0243, "MRR@10": 0.0094},
        "valid": {"NDCG@10": 0.0152, "Hit@10": 0.0285, "MRR@10": 0.0111}
    },
    "Musical_Instruments_ItemCF": {
        "category": "Musical_Instruments", "model": "ItemCF",
        "test": {"NDCG@10": 0.0023, "Hit@10": 0.0044, "MRR@10": 0.0017},
        "valid": {"NDCG@10": 0.0030, "Hit@10": 0.0054, "MRR@10": 0.0023}
    },
    "Musical_Instruments_BPR": {
        "category": "Musical_Instruments", "model": "BPR",
        "test": {"NDCG@10": 0.0211, "Hit@10": 0.0384, "MRR@10": 0.0159},
        "valid": {"NDCG@10": 0.0234, "Hit@10": 0.0427, "MRR@10": 0.0175}
    },
    "CDs_and_Vinyl_BPR": {
        "category": "CDs_and_Vinyl", "model": "BPR",
        "test": {"NDCG@10": 0.0281, "Hit@10": 0.0497, "MRR@10": 0.0215},
        "valid": {"NDCG@10": 0.0321, "Hit@10": 0.0576, "MRR@10": 0.0244}
    }
}

os.makedirs("/root/recsys/results", exist_ok=True)
with open("/root/recsys/results/all_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved!")
print()

print("=" * 70)
print("FINAL RESULTS - Test NDCG@10")
print("=" * 70)
for key in sorted(results.keys()):
    d = results[key]
    print(f"  {d['model']:12s} | {d['category']:25s} | NDCG@10={d['test']['NDCG@10']:.4f}  Hit@10={d['test']['Hit@10']:.4f}")
