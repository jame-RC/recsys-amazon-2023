import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from src.data.dataset import AmazonDataset
    from src.evaluation.evaluator import Evaluator
    from src.models.sasrec import SASRecRecommender
    from src.utils.config import TOP_K, RESULTS_DIR

    cat = "Industrial_and_Scientific"
    dataset = AmazonDataset(cat)
    train_data = dataset.get_train_sequences()
    valid_data = dataset.get_eval_data("valid")
    test_data = dataset.get_eval_data("test")
    num_items = len(dataset.item_vocab)
    print(f"Items: {num_items}, Train: {len(train_data)}", flush=True)

    # Maximize GPU usage on RTX 2060 6GB
    model = SASRecRecommender(
        num_items, 
        embedding_dim=256,        # 4x original
        num_heads=8,              # 4x original
        num_layers=6,             # 3x original
        max_seq_len=50,
        num_epochs=30,
        lr=5e-5,
        batch_size=1024,          # Large batch
        neg_samples=20
    )
    print(f"Device: {model.device}", flush=True)

    model.fit(train_data)

    print("Evaluating valid...", flush=True)
    vm = Evaluator(model, valid_data, num_items, TOP_K).evaluate()
    print(f"Valid: NDCG@10={vm['NDCG@10']:.4f}", flush=True)

    print("Evaluating test...", flush=True)
    tm = Evaluator(model, test_data, num_items, TOP_K).evaluate()
    print(f"Test: NDCG@10={tm['NDCG@10']:.4f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.model.state_dict(), f"{RESULTS_DIR}/Industrial_and_Scientific_SASRec_model.pt")
    import json
    json.dump({"category": cat, "model": "SASRec", "valid": vm, "test": tm},
              open(f"{RESULTS_DIR}/Industrial_and_Scientific_sasrec.json", "w"), indent=2)
    print("Saved!", flush=True)
