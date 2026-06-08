"""Task 9 — Retrieval Pipeline hoàn chỉnh.

Hybrid = semantic cosine + lexical BM25, merge bằng RRF, rerank Jina,
fallback sang PageIndex API.
"""
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank
from .task8_pageindex_vectorless import pageindex_search
from .task6_lexical_search import normalize_text

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "jina"


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    rrf_scores = {}
    content_map = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            key = item.get("content", "")
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map or item.get("score", 0) > content_map[key].get("score", 0):
                content_map[key] = item
    results = []
    for content, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def hyde_expand_query(query: str) -> str:
    """Bonus HyDE (Hypothetical Document Embeddings)."""
    return (
        f"{query}. Tài liệu giả định liên quan pháp luật ma túy, nghệ sĩ Việt Nam, "
        "bị bắt, bị truy tố, tổ chức sử dụng ma túy, tàng trữ, mua bán, nguồn báo chí, "
        "Bộ luật Hình sự, Luật Phòng chống ma túy."
    )


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, score_threshold: float = SCORE_THRESHOLD, use_reranking: bool = True, use_hyde: bool = True) -> list[dict]:
    search_query = hyde_expand_query(query) if use_hyde else query
    search_limit = max(top_k * 6, 20)
    
    dense_results = semantic_search(query, top_k=search_limit) + semantic_search(search_query, top_k=search_limit)
    sparse_results = lexical_search(query, top_k=search_limit) + lexical_search(search_query, top_k=search_limit)
    
    merged = rerank_rrf([dense_results, sparse_results], top_k=max(top_k * 6, 20))
    for item in merged:
        item["source"] = "hybrid"
        
    final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD) if use_reranking and merged else merged[:top_k]

    for item in final_results:
        item.setdefault("metadata", {})["hyde_enabled"] = use_hyde
        item["source"] = "hybrid"
        
    if not final_results or final_results[0].get("score", 0.0) < score_threshold:
        return pageindex_search(query, top_k=top_k)
        
    return final_results[:top_k]

if __name__ == "__main__":
    print(retrieve("hình phạt ma tuý", 3))
