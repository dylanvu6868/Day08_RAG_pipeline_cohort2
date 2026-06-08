"""Task 7 — Reranking Module.

Uses Jina Reranker v2 Base Multilingual API for cross-encoder reranking.
"""

from __future__ import annotations

import os
import requests
from dotenv import load_dotenv

load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "jina") -> list[dict]:
    if not candidates:
        return []

    if not JINA_API_KEY:
        # Fallback to score sorting if no API key
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    # Use Jina API
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # We pass only the content text for Jina to score against the query
    documents = [c.get("content", "") for c in candidates]
    
    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "documents": documents,
        "top_n": top_k
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        
        # results contains: [{"index": 0, "relevance_score": 0.8}, ...]
        reranked = []
        for r in results:
            idx = r["index"]
            item = candidates[idx].copy()
            item["score"] = r["relevance_score"]
            reranked.append(item)
            
        return reranked
    except Exception as e:
        print(f"[Rerank] Jina API error: {e}")
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

if __name__ == "__main__":
    print(rerank("Miu Lê bị bắt", [{"content": "Miu Lê bị giữ để điều tra", "score": 0.2, "metadata": {}}]))
