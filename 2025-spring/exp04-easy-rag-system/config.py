# Milvus Lite Configuration
MILVUS_LITE_DATA_PATH = "./milvus_lite_data.db" # Path to store Milvus Lite data
COLLECTION_NAME = "medical_rag_lite" # Use a different name if needed

# Data Configuration
DATA_FILE = "./data/processed_data.json"

# Model Configuration
# Example: 'all-MiniLM-L6-v2' (dim 384), 'thenlper/gte-large' (dim 1024)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
EMBEDDING_DIM = 384 # Must match EMBEDDING_MODEL_NAME

# Indexing and Search Parameters
MAX_ARTICLES_TO_INDEX = 500
TOP_K = 3
# Milvus index parameters (adjust based on data size and needs)
INDEX_METRIC_TYPE = "L2" # Or "IP"
INDEX_TYPE = "IVF_FLAT"  # Milvus Lite 支持的索引类型
# HNSW index params (adjust as needed)
INDEX_PARAMS = {"nlist": 128}
# HNSW search params (adjust as needed)
SEARCH_PARAMS = {"nprobe": 16}

# Generation Parameters
MAX_NEW_TOKENS_GEN = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Global map to store document content (populated during indexing)
# Key: document ID (int), Value: dict {'title': str, 'abstract': str, 'content': str}
id_to_doc_map = {} 