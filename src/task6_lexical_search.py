"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data" / "standardized"


def load_corpus() -> list[dict]:
    """Tải toàn bộ tài liệu Markdown từ data/standardized/."""
    corpus = []
    if not DATA_DIR.exists():
        return corpus

    for filepath in DATA_DIR.rglob("*.md"):
        text = filepath.read_text(encoding="utf-8")
        if text.strip():
            corpus.append({
                "content": text,
                "metadata": {"source": str(filepath.name), "path": str(filepath)}
            })
    return corpus


CORPUS: list[dict] = load_corpus()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    current_corpus = CORPUS if CORPUS else load_corpus()
    if not current_corpus:
        return []

    tokenized_corpus = [doc["content"].lower().split() for doc in current_corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": current_corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": current_corpus[idx].get("metadata", {})
            })
    return results


if __name__ == "__main__":
    results = lexical_search("LMHT 26.13", top_k=3)
    print(f"Tìm thấy {len(results)} kết quả:")
    for r in results:
        print(f"[{r['score']:.3f}] {r['metadata'].get('source')}")
