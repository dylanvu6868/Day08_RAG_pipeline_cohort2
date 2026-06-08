"""Task 4 — Chunking & Indexing (Weaviate Cloud).

Connects to Weaviate and indexes the chunked documents using OpenAI embeddings.
If Weaviate fails, falls back to local JSON indexing.
"""

from __future__ import annotations

import os
from pathlib import Path
import json
import re
import unicodedata
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_PATH = Path(__file__).parent.parent / "data" / "local_chunks.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# Weaviate Settings
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


BOILERPLATE_PATTERNS = [
    r"Podcast YouTube Cần biết[\s\S]{0,4500}?Email:\s*tto@tuoitre\.com\.vn\s*Hotline:\s*0918\.033\.133\s*0",
    r"Mới nhất Xem nhiều Video Thời sự[\s\S]{0,2500}?Thông tin tòa soạn",
    r"Đặt báo Quảng cáo Email:\s*tto@tuoitre\.com\.vn\s*Hotline:\s*0918\.033\.133\s*0",
]

NOISE_KEYWORDS = {
    "podcast", "youtube", "rao vặt", "đặt báo", "quảng cáo", "hotline",
    "email:", "xem thêm", "mới nhất", "xem nhiều", "thời tiết", "đăng xuất",
    "cài đặt tài khoản", "tin đã lưu", "lịch sử giao dịch",
}


def _norm(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def clean_content(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "\n", text, flags=re.I | re.U)

    cleaned_paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text):
        paragraph = re.split(r"Đọc tiếp|Tin liên quan|Tuổi Trẻ Online Newsletters|Bình luận \(", paragraph, maxsplit=1)[0]
        p = " ".join(paragraph.split()).strip()
        p = re.sub(r"Pháp luật\s+\d{1,2}/\d{1,2}/20\d{2}\s+\d{1,2}:\d{2}\s+GMT\+7\s+[^.]{0,180}?(?=Lê Ánh Nhật|Ngày \d{1,2}-\d{1,2})", "", p, flags=re.I)
        if not p: continue
        low = p.lower()
        noise_hits = sum(1 for kw in NOISE_KEYWORDS if kw in low)
        if noise_hits >= 3 and len(p) > 350: continue
        if len(p) > 1800 and noise_hits >= 1: continue
        if any(noise in low for noise in ["tặng sao", "chủ đề:", "newsletters", "tài khoản được sử dụng", "giấy phép hoạt động báo điện tử"]): continue
        cleaned_paragraphs.append(p)
    return "\n\n".join(cleaned_paragraphs).strip()


def load_documents() -> list[dict]:
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content: continue
        content = clean_content(content)
        if not content: continue
        doc_type = "unknown"
        if "legal" in md_file.parts: doc_type = "legal"
        elif "news" in md_file.parts: doc_type = "news"
        elif "uploads" in md_file.parts: doc_type = "upload"
        
        year_match = re.search(r"(?:Published|Crawled):\*\*\s*(20\d{2})|\b(20\d{2})\b", content[:800])
        year = next((g for g in (year_match.groups() if year_match else []) if g), None)
        title_match = re.search(r"^#\s+(.+)$", content, flags=re.M)
        title = title_match.group(1).strip() if title_match else md_file.stem
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "title": title, "type": doc_type, "path": str(md_file), "year": year},
        })
    return documents


def _sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=\.)\s+", paragraph)
    return [p.strip() for p in parts if p.strip()]


def _split_long_text(text: str) -> list[str]:
    chunks = []; start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        cut = end
        if end < len(text):
            for sep in [". ", "; ", ", ", " "]:
                pos = text.rfind(sep, start, end)
                if pos > start + CHUNK_SIZE // 2:
                    cut = pos + len(sep)
                    break
        chunk = text[start:cut].strip()
        if chunk: chunks.append(chunk)
        if cut >= len(text): break
        start = max(0, cut - CHUNK_OVERLAP)
    return chunks


def _split_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []; current = ""
    for paragraph in paragraphs:
        units = _sentences(paragraph) if len(paragraph) > CHUNK_SIZE else [paragraph]
        for unit in units:
            if len(unit) > CHUNK_SIZE:
                for sub in _split_long_text(unit):
                    if current:
                        chunks.append(current.strip())
                        current = ""
                    chunks.append(sub)
                continue
            candidate = (current + "\n\n" + unit).strip() if current else unit
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
            else:
                if current: chunks.append(current.strip())
                current = unit
    if current: chunks.append(current.strip())
    return chunks


def _title_entity_filter(title: str, chunk_text: str) -> bool:
    title_norm = _norm(title)
    text_norm = _norm(chunk_text)
    known = {
        "miu le": ["miu le", "le anh nhat"],
        "long nhat": ["long nhat", "dinh long nhat"],
        "son ngoc minh": ["son ngoc minh"],
        "chi dan": ["chi dan", "nguyen trung hieu"],
        "an tay": ["an tay", "andrea aybar"],
        "truc phuong": ["truc phuong", "nguyen do truc phuong"],
    }
    title_entities = {entity for entity, aliases in known.items() if any(a in title_norm for a in aliases)}
    if not title_entities: return True
    text_entities = {entity for entity, aliases in known.items() if any(a in text_norm for a in aliases)}
    unrelated = text_entities - title_entities
    if unrelated and not (text_entities & title_entities): return False
    if unrelated and any(marker in text_norm for marker in ["doc tiep", "tin lien quan", "chu de:", "newsletters"]): return False
    return True


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        splits = _split_text(doc["content"])
        filtered_splits = [s for s in splits if _title_entity_filter(metadata.get("title", ""), s)]
        for i, chunk_text in enumerate(filtered_splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**metadata, "chunk_index": i},
            })
    return chunks


def index_to_weaviate(chunks: list[dict]):
    if not WEAVIATE_URL or "xxx" in WEAVIATE_URL or not WEAVIATE_API_KEY:
        print("[Weaviate] Missing valid credentials. Falling back to local.")
        return index_to_local(chunks)
        
    try:
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType
        from weaviate.auth import AuthApiKey

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
        )
        
        # Define collection
        coll_name = "DrugLawChunks"
        if not client.collections.exists(coll_name):
            client.collections.create(
                name=coll_name,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(model="text-embedding-3-small"),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="type", data_type=DataType.TEXT),
                ]
            )
        
        collection = client.collections.get(coll_name)
        with collection.batch.dynamic() as batch:
            for c in chunks:
                md = c.get("metadata", {})
                batch.add_object({
                    "content": c["content"],
                    "source": md.get("source", ""),
                    "title": md.get("title", ""),
                    "type": md.get("type", "")
                })
        
        client.close()
        print("[Weaviate] Uploaded chunks to Weaviate cloud successfully.")
        
    except Exception as e:
        print(f"[Weaviate] Error connecting/uploading: {e}. Fallback to local.")
        index_to_local(chunks)


def index_to_local(chunks: list[dict]):
    INDEX_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return INDEX_PATH


def get_chunks() -> list[dict]:
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    chunks = chunk_documents(load_documents())
    index_to_local(chunks)
    return chunks


def run_pipeline():
    docs = load_documents()
    chunks = chunk_documents(docs)
    index_to_weaviate(chunks)
    
    # Also save local so fallback logic works seamlessly
    index_to_local(chunks)


if __name__ == "__main__":
    run_pipeline()
