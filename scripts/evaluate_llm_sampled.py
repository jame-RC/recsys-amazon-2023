import os
import sys
import time
import json
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AmazonDataset
from src.evaluation.evaluator import Evaluator
from src.llm.api import LLMClient
from src.models.llm_ranker import LLMRankerRecommender
from src.models.bpr import BPRRecommender
from src.utils.config import RESULTS_DIR, TOP_K
from src.utils.logger import get_logger
from src.llm.prompt import get_category_prompts, format_history, format_candidates

logger = get_logger("evaluate_llm_sampled")

class CustomLLMRanker:
    def __init__(self, num_items, base_model, llm_client, item_meta, category, mode="zeroshot"):
        self.num_items = num_items
        self.base_model = base_model
        self.llm_client = llm_client
        self.item_meta = item_meta
        self.category = category
        self.mode = mode
        self.candidate_size = 50
        
        # Load category adapted prompts
        self.system_prompt, self.user_prompt_template = get_category_prompts(category, mode)
        self.name = f"LLM_Ranker_{mode.capitalize()}"

    def _get_item_info(self, item_idx: int) -> dict:
        for asin, meta in self.item_meta.items():
            if meta.get("idx") == item_idx:
                return {"id": item_idx, "title": meta.get("title", f"Item_{item_idx}"),
                        "description": meta.get("description", "")}
        return {"id": item_idx, "title": f"Item_{item_idx}", "description": ""}

    def _build_history_text(self, history: list) -> str:
        items = [self._get_item_info(idx) for idx in history[-10:]]
        return format_history(items)

    def _build_candidate_text(self, candidates: list) -> str:
        items = [{"id": i + 1, "title": self._get_item_info(c)["title"]} for i, c in enumerate(candidates)]
        return format_candidates(items)

    def recommend_single(self, history: list, top_k: int = 10) -> list:
        if not history:
            return []
        
        base_candidates = self.base_model.recommend(history, self.candidate_size)
        if not base_candidates:
            return []

        history_text = self._build_history_text(history)
        candidate_text = self._build_candidate_text(base_candidates)

        prompt = self.user_prompt_template.format(history=history_text, candidates=candidate_text)
        
        # Call DeepSeek API
        ranked_indices = self.llm_client.rank(self.system_prompt, prompt, len(base_candidates))

        result = []
        for idx in ranked_indices:
            if 1 <= idx <= len(base_candidates):
                item = base_candidates[idx - 1]
                if item not in result:
                    result.append(item)
            if len(result) >= top_k:
                break
        return result

def calculate_metrics(recommendations, ground_truth, top_k=10):
    ndcg = 0.0
    hit = 0.0
    mrr = 0.0
    
    if ground_truth in recommendations[:top_k]:
        hit = 1.0
        rank = recommendations[:top_k].index(ground_truth) + 1
        import math
        ndcg = 1.0 / math.log2(rank + 1)
        mrr = 1.0 / rank
        
    return {"NDCG@10": ndcg, "Hit@10": hit, "MRR@10": mrr}

def evaluate_sampled(category, mode, api_key, sample_size=500, max_workers=15):
    print(f"\n==================================================", flush=True)
    print(f" EVALUATING LLM {mode.upper()} ON {category} (Sample: {sample_size})", flush=True)
    print(f"==================================================", flush=True)
    
    dataset = AmazonDataset(category)
    test_data = dataset.get_eval_data("test")
    num_items = len(dataset.item_vocab)
    
    # 1. 随机抽样
    random.seed(42)  # 固定随机种子保证可重复性
    sampled_test_data = random.sample(test_data, min(sample_size, len(test_data)))
    print(f"Sampled {len(sampled_test_data)} users from {len(test_data)} total test users.", flush=True)
    
    # 2. 加载已经训练好的最优 BPR 模型作为候选生成器
    print("Loading pre-trained BPR model as candidate generator...", flush=True)
    bpr_model_path = os.path.join(RESULTS_DIR, f"{category}_BPR_model.pt")
    if os.path.exists(bpr_model_path):
        import torch
        from src.models.bpr import BPRModel
        base_model = BPRRecommender(num_items)
        base_model.model = BPRModel(num_items, base_model.embedding_dim).to(base_model.device)
        base_model.model.load_state_dict(torch.load(bpr_model_path, map_location=base_model.device))
        print("Successfully loaded pre-trained BPR model.", flush=True)
    else:
        print("Warning: BPR model not found! Training a quick BPR on CPU instead...", flush=True)
        train_data = dataset.get_train_sequences()
        base_model = BPRRecommender(num_items)
        base_model.fit(train_data)
        
    # 3. 构造 DeepSeek 客户端
    print("Initializing DeepSeek Client...", flush=True)
    llm_client = LLMClient(provider="openai", model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com")
    
    item_meta = {}
    for asin, meta in dataset.item_meta.items():
        item_meta[asin] = {**meta, "idx": dataset.item_vocab.encode(asin)}
        
    ranker = CustomLLMRanker(num_items, base_model, llm_client, item_meta, category, mode)
    
    # 4. 多线程并发请求，带有指数退避重试逻辑
    print(f"Running LLM inference using {max_workers} threads parallel...", flush=True)
    metrics_sum = {"NDCG@10": 0.0, "Hit@10": 0.0, "MRR@10": 0.0}
    completed_count = 0
    t0 = time.time()
    
    def process_user(user_entry):
        user_idx, history, target = user_entry
        max_retries = 3
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                recs = ranker.recommend_single(history, TOP_K)
                user_metric = calculate_metrics(recs, target, TOP_K)
                return user_metric
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = backoff ** attempt + random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(f"Failed to rank user after {max_retries} attempts: {e}")
                    return {"NDCG@10": 0.0, "Hit@10": 0.0, "MRR@10": 0.0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_user, entry): entry for entry in sampled_test_data}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Querying DeepSeek ({mode})"):
            res = future.result()
            for k in metrics_sum:
                metrics_sum[k] += res[k]
            completed_count += 1

    total_time = time.time() - t0
    avg_metrics = {k: v / completed_count for k, v in metrics_sum.items()}
    avg_metrics["eval_time_sec"] = total_time
    
    print(f"\nResults for {category} ({mode}):", flush=True)
    print(f"  NDCG@10 = {avg_metrics['NDCG@10']:.4f}", flush=True)
    print(f"  Hit@10  = {avg_metrics['Hit@10']:.4f}", flush=True)
    print(f"  MRR@10  = {avg_metrics['MRR@10']:.4f}", flush=True)
    print(f"  Time taken = {total_time:.1f}s", flush=True)
    
    return avg_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=500, help="Number of users to sample")
    parser.add_argument("--max-workers", type=int, default=15, help="Number of parallel worker threads")
    parser.add_argument("--category", type=str, default="all", choices=["all", "Musical_Instruments", "CDs_and_Vinyl", "Industrial_and_Scientific"])
    parser.add_argument("--api-key", type=str, default="", help="DeepSeek API Key (or set DEEPSEEK_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Error: Please provide DeepSeek API key using --api-key or DEEPSEEK_API_KEY environment variable.", flush=True)
        sys.exit(1)
    
    categories = ["Musical_Instruments", "CDs_and_Vinyl", "Industrial_and_Scientific"] if args.category == "all" else [args.category]
    
    for category in categories:
        for mode in ["zeroshot", "fewshot"]:
            res = evaluate_sampled(category, mode, api_key, sample_size=args.sample_size, max_workers=args.max_workers)
            
            # 保存结果到单独的文件
            result_dict = {
                "category": category,
                "model": f"LLM_Ranker_{mode.capitalize() if mode == 'fewshot' else 'ZeroShot'}",
                "valid": res,
                "test": res
            }
            
            filename = f"{category}_llm_ranker_{mode.lower()}.json"
            out_path = os.path.join(RESULTS_DIR, filename)
            with open(out_path, "w") as f:
                json.dump(result_dict, f, indent=2)
            print(f"Successfully saved results to {out_path}!", flush=True)
            
    print("\nEvaluating and syncing complete for all specified categories!")

if __name__ == "__main__":
    main()
