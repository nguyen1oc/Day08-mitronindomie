"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chunking Strategy: Tách chunk theo Tiêu đề Tướng / Chủ đề độc lập (Header-based)
# Không cắt vụn 500 ký tự để giữ nguyên trọn vẹn thông tin từng Tướng
CHUNK_SIZE = 1500        # Ngưỡng kích thước tối đa dành cho fallback split nếu đoạn văn quá dài
CHUNK_OVERLAP = 150      # Độ đè lấp giữa các sub-chunk nếu phải cắt
CHUNKING_METHOD = "markdown_header"  # "recursive" | "markdown_header" | "semantic"

# Embedding Model: nvidia/llama-nemotron-embed-vl-1b-v2:free qua OpenRouter
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_DIM = 2048

# Vector Store: ChromaDB
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

_EMBEDDING_MODEL_CACHE = None


def extract_patch_version(filename: str, content: str = "") -> str:
    """
    Trích xuất thông tin phiên bản bản cập nhật (ví dụ: '26.14', '26.12', '26.13', '26.15').
    """
    match = re.search(r"(\d{2}\.\d{2})", filename)
    if match:
        return match.group(1)
    if content:
        match = re.search(r"(?:Phiên Bản|LMHT|Patch)\s*(\d{2}\.\d{2})", content, re.IGNORECASE)
        if match:
            return match.group(1)
    return "unknown"


class OpenRouterEmbeddingModel:
    def __init__(self, model_name="openai/text-embedding-3-small"):
        self.model_name = model_name
        import os
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set in environment variables! "
                "Please configure it in your .env file."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def encode(self, texts, show_progress_bar=False):
        input_is_string = isinstance(texts, str)
        if input_is_string:
            texts = [texts]
        
        import numpy as np
        
        # Batch size for embeddings to avoid token limits
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float"
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
        embeddings = np.array(all_embeddings)
        if input_is_string:
            return embeddings[0]
        return embeddings


def get_embedding_model():
    """Lazy load embedding model bằng OpenRouter."""
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is None:
        _EMBEDDING_MODEL_CACHE = OpenRouterEmbeddingModel(EMBEDDING_MODEL)
    return _EMBEDDING_MODEL_CACHE


def get_collection():
    """Lấy hoặc tạo ChromaDB collection."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/ và bổ sung patch_version.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, 'patch_version': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.is_file() and not md_file.name.startswith("."):
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in md_file.parts else "news"
            patch_version = extract_patch_version(md_file.name, content)
            documents.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "patch_version": patch_version
                }
            })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents độc lập theo từng Tướng / Chủ đề bằng MarkdownHeaderTextSplitter.
    Mỗi Tướng là 1 Chunk duy nhất trọn vẹn, được gắn context header phiên bản.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    global_chunk_idx = 0

    for doc in documents:
        header_splits = markdown_splitter.split_text(doc["content"])
        patch_ver = doc["metadata"]["patch_version"]
        doc_type = doc["metadata"].get("type", "legal")

        for header_doc in header_splits:
            base_meta = {
                **doc["metadata"],
                **header_doc.metadata
            }

            h1 = str(base_meta.get("Header 1", "")).strip()
            h2 = str(base_meta.get("Header 2", "")).strip()
            h3 = str(base_meta.get("Header 3", "")).strip()

            page_content = header_doc.page_content.strip()
            if not page_content:
                continue

            if doc_type == "legal":
                # Tạo nhãn ngữ cảnh cho tài liệu Legal (Patch Notes)
                if h3:
                    header_prefix = f"[Bản cập nhật {patch_ver} | Mục: {h2} | Tướng/Chủ đề: {h3}]\n"
                    champion = h3
                    section = h2
                elif h2:
                    header_prefix = f"[Bản cập nhật {patch_ver} | Mục: {h2}]\n"
                    champion = ""
                    section = h2
                else:
                    header_prefix = f"[Bản cập nhật {patch_ver}]\n"
                    champion = ""
                    section = ""

                chunk_meta = {
                    "source": base_meta["source"],
                    "type": doc_type,
                    "patch_version": patch_ver,
                    "section": section,
                    "champion": champion,
                }

            else:
                # Tạo nhãn ngữ cảnh cho tài liệu News / Điều khoản / Chính sách
                doc_title = h1 if h1 else base_meta["source"].replace(".md", "").replace("-", " ").title()
                if h3:
                    header_prefix = f"[Tài liệu: {doc_title} | Phần: {h2} | Mục: {h3}]\n"
                    section = f"{h2} > {h3}"
                elif h2:
                    header_prefix = f"[Tài liệu: {doc_title} | Phần: {h2}]\n"
                    section = h2
                elif h1:
                    header_prefix = f"[Tài liệu: {doc_title}]\n"
                    section = h1
                else:
                    header_prefix = f"[Tài liệu: {doc_title}]\n"
                    section = ""

                chunk_meta = {
                    "source": base_meta["source"],
                    "type": doc_type,
                    "patch_version": "N/A",
                    "section": section,
                    "champion": "",
                    "title": doc_title,
                }

            full_text = header_prefix + page_content

            # Nếu độ dài văn bản của section quá lớn (> 3000 chars), mới dùng fallback splitter
            if len(full_text) > 3000:
                sub_splits = fallback_splitter.split_text(full_text)
                for sub_text in sub_splits:
                    cleaned_sub = sub_text.strip()
                    if cleaned_sub:
                        chunks.append({
                            "content": cleaned_sub,
                            "metadata": {
                                **chunk_meta,
                                "chunk_index": global_chunk_idx
                            }
                        })
                        global_chunk_idx += 1
            else:
                chunks.append({
                    "content": full_text,
                    "metadata": {
                        **chunk_meta,
                        "chunk_index": global_chunk_idx
                    }
                })
                global_chunk_idx += 1

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.
    """
    if not chunks:
        return chunks

    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict], reset_db: bool = True):
    """
    Lưu chunks vào vector store ChromaDB (xóa collection cũ để làm sạch rác nếu reset_db=True).
    """
    if not chunks:
        return

    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset_db:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]

    cleaned_metadatas = []
    for c in chunks:
        clean_meta = {}
        for k, v in c["metadata"].items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        cleaned_metadatas.append(clean_meta)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=cleaned_metadatas,
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking Strategy: {CHUNKING_METHOD} (1 Champion/Section = 1 Chunk)")
    print(f"  Embedding Model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[+] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[+] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[+] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks, reset_db=True)
    print("[+] Cleaned old DB & Indexed new chunks to ChromaDB successfully")


if __name__ == "__main__":
    run_pipeline()


