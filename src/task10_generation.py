"""Task 10 — document reordering và sinh câu trả lời có citation."""

import os
import sys

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

# Năm chunks thường đủ evidence nhưng chưa làm prompt quá dài.
TOP_K = 5
# RAG ưu tiên tính ổn định/factual hơn tính sáng tạo.
TOP_P = 0.9
TEMPERATURE = 0.3
# Có thể đổi model trong .env mà không cần sửa source code.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")


SYSTEM_PROMPT = """Bạn là trợ lý phân tích tài liệu về game: bản cập nhật/patch notes,
chính sách, điều khoản, luật, bảo mật, cơ chế và thuật ngữ trong game.

Quy tắc bắt buộc:
1. Mở đầu bằng "Câu hỏi được hiểu là: ..." và nhắc lại đầy đủ ý định của người dùng.
2. Chỉ sử dụng thông tin trong context; không suy đoán hoặc bịa đặt.
3. Mỗi khẳng định phải có citation ngay sau nó theo đúng dạng [Tên nguồn tài liệu].
4. Chỉ dùng tên nguồn xuất hiện trong nhãn "Nguồn tài liệu" của context.
5. Nếu context không đủ bằng chứng, trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có".
6. Trả lời bằng tiếng Việt, rõ ràng và trực tiếp vào câu hỏi."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Xếp chunks theo ``front + back[::-1]`` để giảm lost-in-the-middle.

    Input theo score: [1, 2, 3, 4, 5]
    Output:           [1, 3, 5, 4, 2]
    """
    # Không mutate list đầu vào để ranking gốc vẫn dùng được ở UI/citation.
    if len(chunks) <= 2:
        return list(chunks)

    # Rank lẻ nằm ở đầu; rank chẵn được đảo và nằm cuối. Vì vậy hai chunks
    # quan trọng nhất nằm ở hai vùng LLM chú ý mạnh nhất: đầu và cuối context.
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Gắn nhãn nguồn vào từng chunk để LLM tạo citation kiểm chứng được."""
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}

        # Dense/BM25 thường có source; PageIndex có thể chỉ có section/doc_id.
        # Fallback cuối bảo đảm không chunk nào thiếu nhãn citation.
        source = (
            metadata.get("source")
            or metadata.get("section")
            or metadata.get("doc_id")
            or f"Nguồn {index}"
        )
        document_type = metadata.get("type", chunk.get("source", "unknown"))
        content = str(chunk.get("content", "")).strip()
        context_parts.append(
            f"[Tài liệu {index} | Nguồn tài liệu: {source} | Loại: {document_type}]\n"
            f"{content}"
        )

    # Phân cách rõ để LLM không trộn nội dung và nguồn giữa hai tài liệu.
    return "\n\n---\n\n".join(context_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Chạy retrieval → reorder → format context → LLM generation có citation."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k phải là số nguyên dương")

    # Bước 1: lấy evidence từ hybrid retrieval hoặc PageIndex fallback (Task 9).
    chunks = retrieve(query.strip(), top_k=top_k)
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    # Không có evidence thì không gọi LLM, tránh hallucination và tốn API.
    if not chunks:
        return {
            "answer": (
                f"Câu hỏi được hiểu là: {query.strip()}\n\n"
                "Tôi không thể xác minh thông tin này từ nguồn hiện có."
            ),
            "sources": [],
            "retrieval_source": "none",
        }

    # Bước 2–3: chống lost-in-the-middle và gắn nhãn nguồn cho citation.
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = (
        f"Context:\n{context}\n\n---\n\n"
        f"Câu hỏi người dùng: {query.strip()}\n\n"
        "Hãy nhắc lại cách bạn hiểu câu hỏi trước, sau đó trả lời với citation."
    )

    # Bước 4: OpenRouter tương thích OpenAI SDK. Nếu dùng OpenAI trực tiếp thì
    # bỏ namespace "openai/" khỏi model mặc định.
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    api_key = openrouter_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env")
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1" if openrouter_key else None,
    )
    model = LLM_MODEL if openrouter_key else LLM_MODEL.removeprefix("openai/")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("LLM không trả về nội dung")

    # Trả chunks theo ranking retrieval gốc để UI hiển thị score đúng.
    return {
        "answer": answer.strip(),
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    # Windows PowerShell có thể dùng cp1252; chuyển UTF-8 để in tiếng Việt.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "Tóm tắt thay đổi quan trọng trong bản cập nhật LMHT 26.15.",
        "Riot xử lý dữ liệu cá nhân của người chơi như thế nào?",
        "Giải thích thuật ngữ sát thương chuẩn trong game.",
    ]
    for question in test_queries:
        print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")
        result = generate_with_citation(question)
        print(f"\nA: {result['answer']}")
        print(
            f"\n[Sources: {len(result['sources'])} chunks | "
            f"via {result['retrieval_source']}]"
        )
