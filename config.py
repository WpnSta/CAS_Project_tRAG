# config.py — shared parameters for all RAG notebooks

EMBED_MODEL      = "intfloat/multilingual-e5-small"
CE_MODEL         = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BATCH_SIZE       = 64

# Retrieval
TOP_K         = 2
BM25_WEIGHT   = 0.4
SEMANTIC_POOL = 30
RERANK_POOL   = 6

# Generation
GENERATION_MODEL = "gemma3:4b"
JUDGE_MODEL      = "phi4-mini"
