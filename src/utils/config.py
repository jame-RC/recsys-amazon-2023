import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CATEGORIES = [
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "CDs_and_Vinyl",
]

DATA_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core"
LEAVE_LAST_OUT_URL = f"{DATA_BASE_URL}/last_out_w_his"
META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"

DEFAULT_CATEGORY = "Industrial_and_Scientific"
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
NUM_EPOCHS = 100
PATIENCE = 10
NEG_SAMPLES = 100
TOP_K = 10
LLM_CANDIDATE_SIZE = 50
