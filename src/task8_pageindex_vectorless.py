"""Task 8 — PageIndex vectorless retrieval fallback.

PageIndex navigates a document's tree hierarchy instead of searching vectorized
chunks. The cloud retrieval endpoint is used here because it returns the raw
relevant sections required by the rest of the RAG pipeline.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_API_URL = os.getenv("PAGEINDEX_API_URL", "https://api.pageindex.ai").rstrip("/")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
POLL_INTERVAL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 60.0
PROCESS_TIMEOUT_SECONDS = float(os.getenv("PAGEINDEX_PROCESS_TIMEOUT", "600"))


def _headers() -> dict[str, str]:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được cấu hình trong file .env")
    return {"api_key": PAGEINDEX_API_KEY}


def _markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    """Convert one UTF-8 Markdown document to a searchable temporary PDF."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise RuntimeError(
            "Cần cài fpdf2 để chuyển tài liệu standardized sang PDF: pip install fpdf2"
        ) from exc

    font_path = Path(os.getenv("PAGEINDEX_PDF_FONT", r"C:\Windows\Fonts\arial.ttf"))
    if not font_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy Unicode font. Hãy set PAGEINDEX_PDF_FONT tới file .ttf."
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DocumentFont", fname=str(font_path))
    pdf.add_page()
    pdf.set_font("DocumentFont", size=10)
    text = markdown_path.read_text(encoding="utf-8")
    pdf.multi_cell(w=0, h=5, text=text, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(pdf_path))


def _wait_until_processed(doc_id: str) -> dict:
    """Poll PageIndex until its hierarchy is built and retrieval is ready."""
    deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(
            f"{PAGEINDEX_API_URL}/doc/{doc_id}/",
            headers=_headers(),
            params={"type": "tree", "summary": "true"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "")).lower()
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(
                f"PageIndex xử lý document {doc_id} thất bại: "
                f"{payload.get('error') or status}"
            )
        # Chat API hiện hành sử dụng được tree ngay khi processing completed.
        # Không chờ retrieval_ready vì field đó chỉ dành cho Retrieval API legacy
        # và có thể luôn là False dù tree đã được tạo thành công.
        if status in {"completed", "success", "succeeded"} and payload.get("result"):
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"PageIndex xử lý document {doc_id} quá thời gian chờ")


def upload_documents() -> list[dict]:
    """Upload và process tài liệu PDF/Markdown trong ``data/landing``.

    PDF được upload trực tiếp. Markdown được chuyển thành PDF trong thư mục tạm.
    Sau upload, hàm chờ PageIndex xây xong tree và chỉ trả về khi document đã
    sẵn sàng cho vectorless retrieval.
    """
    source_files = (
        sorted(
            path
            for path in LANDING_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".md"}
        )
        if LANDING_DIR.exists()
        else []
    )
    if not source_files:
        raise FileNotFoundError(
            f"Không tìm thấy tài liệu PDF hoặc Markdown trong {LANDING_DIR}"
        )

    existing_response = requests.get(
        f"{PAGEINDEX_API_URL}/docs",
        headers=_headers(),
        params={"limit": 100, "offset": 0},
        timeout=30,
    )
    existing_response.raise_for_status()
    existing_by_name = {
        document.get("name"): document.get("id") or document.get("doc_id")
        for document in existing_response.json().get("documents", [])
        if document.get("name") and (document.get("id") or document.get("doc_id"))
    }

    uploaded = []
    with tempfile.TemporaryDirectory(prefix="pageindex_upload_") as temp_dir:
        temp_path = Path(temp_dir)
        for source_path in source_files:
            if source_path.suffix.lower() == ".pdf":
                pdf_path = source_path
            else:
                relative_parent = source_path.parent.relative_to(LANDING_DIR)
                prefix = "_".join(relative_parent.parts)
                pdf_name = f"{prefix}_{source_path.stem}.pdf" if prefix else f"{source_path.stem}.pdf"
                pdf_path = temp_path / pdf_name
                _markdown_to_pdf(source_path, pdf_path)

            doc_id = existing_by_name.get(pdf_path.name)
            if doc_id:
                print(f"  Reusing: {source_path.name} -> {doc_id}", flush=True)
            else:
                with pdf_path.open("rb") as file_obj:
                    response = requests.post(
                        f"{PAGEINDEX_API_URL}/doc/",
                        headers=_headers(),
                        files={"file": (pdf_path.name, file_obj, "application/pdf")},
                        timeout=120,
                    )
                response.raise_for_status()
                payload = response.json()
                doc_id = payload.get("doc_id") or payload.get("id")
                if not doc_id:
                    raise RuntimeError(
                        f"PageIndex không trả về doc_id cho {source_path.name}"
                    )
                print(f"  Uploaded: {source_path.name} -> {doc_id}", flush=True)

            processed = _wait_until_processed(doc_id)
            uploaded.append(
                {
                    "doc_id": doc_id,
                    "source": str(source_path.relative_to(LANDING_DIR)),
                    "input_type": source_path.suffix.lower().lstrip("."),
                    "status": processed.get("status", "completed"),
                    "retrieval_ready": processed.get("retrieval_ready", True),
                }
            )
            print(
                f"  Processed: {source_path.name} -> retrieval ready",
                flush=True,
            )

    return uploaded


def _document_ids() -> list[str]:
    """Lấy doc IDs từ cấu hình hoặc danh sách tài liệu trên PageIndex account."""
    configured = os.getenv("PAGEINDEX_DOC_IDS", "")
    if configured.strip():
        return [item.strip() for item in configured.split(",") if item.strip()]

    response = requests.get(
        f"{PAGEINDEX_API_URL}/docs",
        headers=_headers(),
        params={"limit": 100, "offset": 0},
        timeout=30,
    )
    response.raise_for_status()
    documents = response.json().get("documents", [])
    return [
        str(document.get("id") or document.get("doc_id"))
        for document in documents
        if (document.get("id") or document.get("doc_id"))
        and document.get("status", "completed") == "completed"
    ]


def _retrieve_document(doc_id: str, query: str) -> dict:
    """Submit one tree retrieval task and poll until it reaches a terminal state."""
    response = requests.post(
        f"{PAGEINDEX_API_URL}/retrieval/",
        headers=_headers(),
        json={"doc_id": doc_id, "query": query, "thinking": True},
        timeout=30,
    )
    response.raise_for_status()
    submitted = response.json()
    retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
    if not retrieval_id:
        raise RuntimeError(f"PageIndex không trả về retrieval_id cho document {doc_id}")

    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status_response = requests.get(
            f"{PAGEINDEX_API_URL}/retrieval/{retrieval_id}/",
            headers=_headers(),
            timeout=30,
        )
        status_response.raise_for_status()
        retrieval = status_response.json()
        status = str(retrieval.get("status", "")).lower()
        if status in {"completed", "success", "succeeded"}:
            return retrieval
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(
                f"PageIndex retrieval {retrieval_id} thất bại: "
                f"{retrieval.get('error') or status}"
            )
        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"PageIndex retrieval {retrieval_id} vượt quá thời gian chờ")


def _content_items(relevant_contents) -> list[dict]:
    """Flatten both current and older nested ``relevant_contents`` schemas."""
    if isinstance(relevant_contents, dict):
        return [relevant_contents]
    if not isinstance(relevant_contents, list):
        return []

    flattened = []
    for item in relevant_contents:
        flattened.extend(_content_items(item))
    return flattened


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve context bằng PageIndex Chat API dựa trên document tree.

    Retrieval API cũ có thể trả ``completed`` nhưng ``retrieved_nodes=[]`` khi
    ``retrieval_ready`` không còn được bật. Chat API là endpoint hiện hành và tự
    điều hướng tree để lấy context, không dùng vector database hoặc chunking.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k phải là số nguyên dương")

    doc_ids = _document_ids()
    if not doc_ids:
        raise RuntimeError("Không có PageIndex document đã process để truy vấn")

    response = requests.post(
        f"{PAGEINDEX_API_URL}/chat/completions",
        headers={**_headers(), "Content-Type": "application/json"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Dùng cấu trúc cây của tài liệu để tìm đầy đủ nội dung liên quan. "
                        "Chỉ trả lời dựa trên tài liệu, giữ citation theo trang và nói rõ "
                        "nếu tài liệu không có đủ bằng chứng.\n\n"
                        f"Câu hỏi: {query.strip()}"
                    ),
                }
            ],
            "doc_id": doc_ids,
            "stream": False,
            "enable_citations": True,
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    content = (
        choices[0].get("message", {}).get("content", "") if choices else ""
    )
    if not content or not content.strip():
        raise RuntimeError("PageIndex Chat API không trả về nội dung")

    # Chat API trả một context/answer tổng hợp có citation. Score=1 biểu thị kết
    # quả PageIndex duy nhất theo rank, không phải cosine similarity.
    return [
        {
            "content": content.strip(),
            "score": 1.0,
            "metadata": {
                "doc_ids": doc_ids,
                "type": "pageindex_tree_chat",
                "citations_enabled": True,
            },
            "source": "pageindex",
        }
    ][:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
    else:
        for result in pageindex_search(
            "Tóm tắt những thay đổi quan trọng trong bản cập nhật game",
            top_k=3,
        ):
            print(f"[{result['score']:.3f}] {result['content'][:100]}...")
