"""Task 5 — Semantic Search Module.

Uses Weaviate Cloud (with OpenAI embeddings) for semantic search.
Falls back to local TF-IDF if Weaviate is unavailable.
"""

from __future__ import annotations

import os
from collections import Counter
import math
import re
import unicodedata
from dotenv import load_dotenv
from .task4_chunking_indexing import get_chunks
from .task6_lexical_search import tokenize

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

_CORPUS = None
_DOC_VECS = None
_IDF = None


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return text


def _tokens(text: str) -> list[str]:
    return tokenize(text) + re.findall(r"\w+", normalize_text(text), re.UNICODE)


def _build_index():
    global _CORPUS, _DOC_VECS, _IDF
    _CORPUS = get_chunks()
    tokenized = [_tokens(c["content"]) for c in _CORPUS]
    N = len(tokenized)
    df = {}
    for toks in tokenized:
        for token in set(toks):
            df[token] = df.get(token, 0) + 1
    _IDF = {token: math.log((N + 1) / (freq + 1)) + 1 for token, freq in df.items()}
    _DOC_VECS = []
    for toks in tokenized:
        cnt = Counter(toks)
        vec = {token: (1 + math.log(tf)) * _IDF.get(token, 1.0) for token, tf in cnt.items()}
        _DOC_VECS.append(vec)


def _cos(a, b):
    if not a or not b: return 0.0
    dot = sum(value * b.get(token, 0.0) for token, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _query_vec(query):
    cnt = Counter(_tokens(query))
    return {token: (1 + math.log(tf)) * (_IDF.get(token, 1.0) if _IDF else 1.0) for token, tf in cnt.items()}


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    if WEAVIATE_URL and WEAVIATE_API_KEY and "xxx" not in WEAVIATE_URL:
        try:
            import weaviate
            from weaviate.auth import AuthApiKey
            
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
                headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
            )
            coll_name = "DrugLawChunks"
            if client.collections.exists(coll_name):
                collection = client.collections.get(coll_name)
                response = collection.query.near_text(
                    query=query,
                    limit=top_k,
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
                )
                
                results = []
                for obj in response.objects:
                    # Convert distance to a similarity score (approx)
                    dist = obj.metadata.distance if obj.metadata.distance is not None else 0.5
                    score = max(0, 1 - dist)
                    results.append({
                        "content": obj.properties.get("content", ""),
                        "score": score,
                        "metadata": {
                            "source": obj.properties.get("source", ""),
                            "title": obj.properties.get("title", ""),
                            "type": obj.properties.get("type", "")
                        }
                    })
                client.close()
                if results:
                    return results
        except Exception as e:
            print(f"[Semantic] Weaviate search failed: {e}. Falling back to local TF-IDF.")

    # Local fallback
    global _CORPUS
    if _CORPUS is None:
        _build_index()
    if not _CORPUS:
        return []
    qv = _query_vec(query)
    scores = [_cos(qv, doc_vec) for doc_vec in _DOC_VECS]
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"content": _CORPUS[i]["content"], "score": float(scores[i]), "metadata": _CORPUS[i].get("metadata", {})}
        for i in idxs
    ]


if __name__ == "__main__":
    print(semantic_search("hình phạt ma tuý", 3))
