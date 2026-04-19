"""
Three-pass retrieval pipeline (semantic → BM25 hybrid → cross-encoder rerank).
Imported by all RAG notebooks.
"""

import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list:
    """Normalize punctuation to spaces, lowercase, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_bm25_index(chunk_records: list) -> BM25Okapi:
    """
    Build a BM25Okapi index from chunk_records.
    Position i in the corpus corresponds exactly to chunk_records[i].
    ChromaDB IDs must be str(i) to maintain this bijection.
    """
    corpus = [_tokenize(r["text"]) for r in chunk_records]
    return BM25Okapi(corpus)


def retrieve(query, embedder, cross_encoder, collection, bm25_index, chunk_records, cfg):
    """
    Three-pass retrieval returning the top-K most relevant chunks.

    Parameters
    ----------
    query         : str
    embedder      : SentenceTransformer  (intfloat/multilingual-e5-small)
    cross_encoder : CrossEncoder         (mmarco-mMiniLMv2-L12-H384-v1)
    collection    : chromadb.Collection  (cosine distance space)
    bm25_index    : BM25Okapi
    chunk_records : list[dict]           (same order as BM25 corpus)
    cfg           : dict with keys TOP_K, BM25_WEIGHT, SEMANTIC_POOL, RERANK_POOL

    Returns
    -------
    list[dict] — top TOP_K results, each dict contains all chunk metadata plus:
        sem_score, bm25_score, hybrid_score, rerank_score, score (= rerank_score)
    """
    TOP_K         = cfg["TOP_K"]
    BM25_WEIGHT   = cfg["BM25_WEIGHT"]
    SEMANTIC_POOL = cfg.get("SEMANTIC_POOL", TOP_K * 10)
    RERANK_POOL   = cfg.get("RERANK_POOL",   TOP_K * 2)

    n_available = collection.count()
    if n_available == 0:
        raise RuntimeError("ChromaDB collection is empty — run the indexing cell first.")

    # ------------------------------------------------------------------
    # Pass 1: semantic retrieval via ChromaDB HNSW
    # ------------------------------------------------------------------
    q_vec = embedder.encode(["query: " + query], normalize_embeddings=True)
    n_results = min(SEMANTIC_POOL, n_available)

    results      = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=n_results,
    )
    pool_ids       = [int(id_) for id_ in results["ids"][0]]
    pool_distances = results["distances"][0]
    pool_texts     = results["documents"][0]
    pool_metas     = results["metadatas"][0]

    # cosine distance → similarity
    sem_scores = [1.0 - d for d in pool_distances]

    # ------------------------------------------------------------------
    # Pass 2: BM25 hybrid scoring
    # ------------------------------------------------------------------
    tok_query = _tokenize(query)
    raw_bm25  = bm25_index.get_scores(tok_query)
    pool_bm25 = [float(raw_bm25[idx]) for idx in pool_ids]

    bm25_max = max(pool_bm25) if pool_bm25 else 1.0
    bm25_min = min(pool_bm25) if pool_bm25 else 0.0
    if bm25_max == bm25_min:
        bm25_norm = [0.5] * len(pool_bm25)
    else:
        bm25_norm = [(s - bm25_min) / (bm25_max - bm25_min) for s in pool_bm25]

    hybrid_scores = [
        (1.0 - BM25_WEIGHT) * sem + BM25_WEIGHT * bm25
        for sem, bm25 in zip(sem_scores, bm25_norm)
    ]

    # Keep top RERANK_POOL by hybrid score
    pool = list(zip(pool_ids, pool_texts, pool_metas,
                    sem_scores, bm25_norm, hybrid_scores))
    pool.sort(key=lambda x: x[5], reverse=True)
    pool = pool[:RERANK_POOL]

    # ------------------------------------------------------------------
    # Pass 3: cross-encoder reranking
    # ------------------------------------------------------------------
    pairs     = [[query, item[1]] for item in pool]
    ce_scores = cross_encoder.predict(pairs)

    final = []
    for j, item in enumerate(pool):
        meta = dict(item[2])          # copy metadata from ChromaDB
        meta["text"]          = item[1]
        meta["sem_score"]     = item[3]
        meta["bm25_score"]    = item[4]
        meta["hybrid_score"]  = item[5]
        meta["rerank_score"]  = float(ce_scores[j])
        meta["score"]         = float(ce_scores[j])
        final.append(meta)

    final.sort(key=lambda x: x["rerank_score"], reverse=True)
    return final[:TOP_K]
