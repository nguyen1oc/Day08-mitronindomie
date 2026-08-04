"""Task 5 — Semantic Dense Search với Cosine Similarity và HyDE."""

import os

from dotenv import load_dotenv

load_dotenv()


def _generate_hypothetical_document(query: str) -> str:
    """Sinh hypothetical document cho HyDE; fallback về query nếu LLM lỗi."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return query

    try:
        from openai import OpenAI

        use_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1" if use_openrouter else None,
        )
        response = client.chat.completions.create(
            model=os.getenv(
                "HYDE_MODEL",
                "openai/gpt-4o-mini" if use_openrouter else "gpt-4o-mini",
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là chuyên gia phân tích tài liệu liên quan đến game LoL, bao gồm "
                        "bản cập nhật và patch notes, tư vấn về chính sách và luật, cũng như "
                        "giải thích thuật ngữ, cơ chế và nội dung trong game. Hãy tạo một "
                        "đoạn trả lời giả định ngắn, đầy đủ ngữ nghĩa. "
                        "Bắt buộc mở đầu bằng 'Câu hỏi của người dùng là: ...' và diễn đạt lại "
                        "đầy đủ câu hỏi của người dùng trước khi trả lời. Không được bỏ sót "
                        "intent, điều kiện, tên game, phiên bản, tên riêng, mã, con số hoặc "
                        "thuật ngữ quan trọng trong câu hỏi. Sau đó mới viết nội dung trả lời "
                        "giả định. Không thêm thông tin ngoài hai phần này."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=250,
        )
        hypothetical_document = response.choices[0].message.content
        return hypothetical_document.strip() if hypothetical_document else query
    except Exception:
        # HyDE chỉ hỗ trợ retrieval; lỗi LLM không được làm hỏng dense search.
        return query


def semantic_search(
    query: str,
    top_k: int = 10,
    use_hyde: bool = True,
) -> list[dict]:
    """Tìm các chunk gần nhất bằng cosine similarity.

    Args:
        query: Câu truy vấn của người dùng.
        top_k: Số kết quả tối đa.
        use_hyde: Sinh hypothetical document trước khi embedding nếu True.

    Returns:
        Danh sách ``content``, ``score`` và ``metadata``, sắp xếp giảm dần.
    """
    # Task 4 chịu trách nhiệm cung cấp embedding model và vector collection.
    from .task4_chunking_indexing import get_collection, get_embedding_model

    retrieval_query = _generate_hypothetical_document(query) if use_hyde else query
    query_vector = get_embedding_model().encode(retrieval_query).tolist()

    results = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Chroma cosine distance -> cosine similarity.
        score = max(0.0, 1.0 - float(distance))
        output.append(
            {
                "content": document,
                "score": round(score, 4),
                "metadata": metadata,
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    for result in semantic_search("Bộ kĩ năng của Locke", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
