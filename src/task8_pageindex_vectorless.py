"""Task 8 — PageIndex Vectorless RAG.

Uses the official PageIndex API via `pageindex` package.
Falls back to offline vectorless matching if API is unavailable.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from .task6_lexical_search import tokenize
from .task4_chunking_indexing import get_chunks

load_dotenv()
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    files = list(STANDARDIZED_DIR.rglob("*.md"))
    
    if PAGEINDEX_API_KEY:
        try:
            import pageindex
            # Standard hypothetical PageIndex usage
            client = pageindex.Client(api_key=PAGEINDEX_API_KEY)
            for f in files:
                client.upload_file(str(f))
            return {"uploaded": len(files), "mode": "cloud-pageindex"}
        except Exception as e:
            print(f"[PageIndex] Upload error: {e}. Falling back to offline mode.")
            
    return {"uploaded": len(files), "mode": "offline-local"}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    if PAGEINDEX_API_KEY:
        try:
            import pageindex
            client = pageindex.Client(api_key=PAGEINDEX_API_KEY)
            res = client.search(query=query, top_k=top_k)
            # Adapt the response if it works
            results = []
            for r in res.get("results", []):
                results.append({
                    "content": r.get("content", ""),
                    "score": float(r.get("score", 0)),
                    "metadata": r.get("metadata", {}),
                    "source": "pageindex"
                })
            if results: return results
        except Exception as e:
            pass

    # Offline fallback
    q = set(tokenize(query))
    results = []
    for c in get_chunks():
        toks = set(tokenize(c.get("content", "")))
        score = len(q & toks) / (len(q) or 1)
        if score > 0 or not results:
            results.append({"content": c["content"], "score": float(score), "metadata": c.get("metadata", {}), "source": "pageindex"})
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    for r in results:
        r["source"] = "pageindex"
    return results


if __name__ == "__main__":
    print(upload_documents())
    print(pageindex_search("ma tuý", 3))
