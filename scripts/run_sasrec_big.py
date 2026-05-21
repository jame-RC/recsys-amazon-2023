import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from src.data.dataset import AmazonDataset
    from src.evaluation.evaluator import Evaluator
    from src.models.sasrec import SASRecRecommender
    from src.utils.config import TOP_K, RESULTS_DIR

    cat = "Industrial_and_Scientific"
    d = AmazonDataset(cat)
    td = d.get_train_sequences()
    vd = d.get_eval_data("valid")
    ted = d.get_eval_data("test")
    num_items = len(d.item_vocab)
    print(f"Items={num_items} Train={len(td)}", flush=True)

    m = SASRecRecommender(num_items, embedding_dim=256, num_heads=4,
                          num_layers=6, max_seq_len=50, num_epochs=20,
                          lr=5e-5, batch_size=512, neg_samples=20)
    print(f"Device: {m.device}", flush=True)

    m.fit(td)

    print("Valid...", flush=True)
    v = Evaluator(m, vd, num_items, TOP_K).evaluate()
    print(f"Valid NDCG={v['NDCG@10']:.4f}", flush=True)

    print("Test...", flush=True)
    t = Evaluator(m, ted, num_items, TOP_K).evaluate()
    print(f"Test NDCG={t['NDCG@10']:.4f}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(m.model.state_dict(), os.path.join(RESULTS_DIR, "SASRec_Industrial.pt"))
    import json
    json.dump({"valid": v, "test": t}, open(os.path.join(RESULTS_DIR, "sasrec_result.json"), "w"), indent=2)
    print("Done!", flush=True)
