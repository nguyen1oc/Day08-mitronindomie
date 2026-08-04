"""Streamlit UI để phân tích tài liệu game và so sánh Baseline/Advanced RAG."""

from __future__ import annotations

import os
import logging
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Chi tiết exception chỉ xuất hiện trong server log, tuyệt đối không trả ra UI.
logger = logging.getLogger("gamedocs_rag")

ERROR_CATALOG = {
    "RAG-RET-101": "Module retrieval chưa sẵn sàng.",
    "RAG-RET-102": "Pipeline retrieval chưa được triển khai đầy đủ.",
    "RAG-RET-103": "Không thể truy vấn nguồn tài liệu.",
    "RAG-AUTH-101": "Thiếu hoặc sai cấu hình API key.",
    "RAG-GEN-101": "Không thể sinh câu trả lời từ mô hình.",
    "RAG-SYS-100": "Hệ thống gặp lỗi không xác định.",
}

st.set_page_config(
    page_title="GameDocs RAG Lab",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1450px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {background:#141923; border:1px solid #293244;
      border-radius:12px; padding:12px;}
    .pipeline-card {background:#111722; border:1px solid #293244;
      border-radius:12px; padding:14px; margin-bottom:12px;}
    .baseline {border-left:4px solid #7c8da6;}
    .advanced {border-left:4px solid #ff4655;}
    .small-muted {color:#9ba8b8; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


BASELINE_CONFIG = {
    "use_hyde": False,
    "use_bm25": False,
    "use_rrf": False,
    "use_pageindex": False,
    "use_reorder": False,
}


def _source_name(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata") or {}
    return str(
        metadata.get("source")
        or metadata.get("section")
        or metadata.get("doc_id")
        or f"Nguồn {index}"
    )


def _deduplicate(results: list[dict], top_k: int) -> list[dict]:
    unique = []
    seen = set()
    for item in results:
        key = item.get("content", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:top_k]


def retrieve_with_config(query: str, top_k: int, config: dict) -> tuple[list[dict], str]:
    """Chạy retrieval theo config UI để A/B test từng cải tiến độc lập."""
    from src.task5_semantic_search import semantic_search

    dense = semantic_search(query, top_k=top_k * 2, use_hyde=config["use_hyde"])
    best_dense_score = dense[0]["score"] if dense else 0.0
    results = dense
    retrieval_label = "Dense cosine + HyDE" if config["use_hyde"] else "Dense cosine"

    if config["use_bm25"]:
        from src.task6_lexical_search import lexical_search

        sparse = lexical_search(query, top_k=top_k * 2)
        if config["use_rrf"]:
            from src.task7_reranking import rerank_rrf

            results = rerank_rrf([dense, sparse], top_k=top_k * 2)
            retrieval_label += " + BM25 + RRF"
        else:
            # Không cộng trực tiếp cosine và BM25 vì khác thang điểm.
            results = _deduplicate(dense + sparse, top_k * 2)
            retrieval_label += " + BM25 (no fusion)"

    if config["use_pageindex"] and best_dense_score < config["score_threshold"]:
        from src.task8_pageindex_vectorless import pageindex_search

        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback, "PageIndex fallback"

    for item in results:
        item.setdefault("source", "hybrid" if config["use_bm25"] else "dense")
    return results[:top_k], retrieval_label


def generate_answer(query: str, chunks: list[dict], use_reorder: bool) -> str:
    """Sinh answer từ đúng tập chunks của mỗi nhánh A/B để so sánh công bằng."""
    from openai import OpenAI
    from src.task10_generation import (
        LLM_MODEL,
        SYSTEM_PROMPT,
        TEMPERATURE,
        TOP_P,
        format_context,
        reorder_for_llm,
    )

    if not chunks:
        return (
            f"Câu hỏi được hiểu là: {query}\n\n"
            "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        )

    prompt_chunks = reorder_for_llm(chunks) if use_reorder else list(chunks)
    context = format_context(prompt_chunks)
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
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}\n"
                    "Nhắc lại cách hiểu câu hỏi, sau đó trả lời kèm citation."
                ),
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return (response.choices[0].message.content or "").strip()


def run_pipeline(query: str, top_k: int, config: dict) -> dict:
    started = time.perf_counter()
    stage = "retrieval"
    try:
        chunks, retrieval_label = retrieve_with_config(query, top_k, config)
        retrieval_finished = time.perf_counter()
        stage = "generation"
        answer = generate_answer(query, chunks, config["use_reorder"])
        finished = time.perf_counter()
        return {
            "answer": answer,
            "sources": chunks,
            "retrieval": retrieval_label,
            "retrieval_ms": (retrieval_finished - started) * 1000,
            "total_ms": (finished - started) * 1000,
            "error": None,
        }
    except Exception as exc:
        finished = time.perf_counter()

        # Chuyển exception nội bộ thành mã lỗi công khai. Không đưa str(exc),
        # traceback, module hoặc filesystem path vào dữ liệu render trên UI.
        error_text = str(exc).lower()
        if isinstance(exc, ImportError):
            error_code = "RAG-RET-101"
        elif isinstance(exc, NotImplementedError):
            error_code = "RAG-RET-102"
        elif "api_key" in error_text or "api key" in error_text:
            error_code = "RAG-AUTH-101"
        elif stage == "retrieval":
            error_code = "RAG-RET-103"
        elif stage == "generation":
            error_code = "RAG-GEN-101"
        else:
            error_code = "RAG-SYS-100"

        # exc_info=True giữ traceback trong terminal/server log để developer tra.
        logger.exception(
            "Pipeline failed with public_code=%s stage=%s",
            error_code,
            stage,
        )
        return {
            "answer": "Pipeline chưa thể tạo câu trả lời.",
            "sources": [],
            "retrieval": "Unavailable",
            "retrieval_ms": 0.0,
            "total_ms": (finished - started) * 1000,
            "error": {
                "code": error_code,
                "message": ERROR_CATALOG[error_code],
            },
        }


def render_sources(sources: list[dict], key: str) -> None:
    if not sources:
        st.caption("Không có tài liệu retrieval.")
        return
    with st.expander(f"📚 Evidence ({len(sources)} tài liệu)", expanded=False):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            score = float(source.get("score", 0.0))
            st.markdown(
                f"**[{index}] {_source_name(source, index)}**  ·  "
                f"score `{score:.4f}`  ·  `{source.get('source', metadata.get('type', 'unknown'))}`"
            )
            st.caption(str(source.get("content", ""))[:500])
            if index != len(sources):
                st.divider()


def render_result(title: str, result: dict, accent: str, key: str) -> None:
    st.markdown(f"### {accent} {title}")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Retrieval", f"{result['retrieval_ms']:.0f} ms")
    metric_columns[1].metric("Tổng thời gian", f"{result['total_ms'] / 1000:.2f} s")
    metric_columns[2].metric("Evidence", len(result["sources"]))
    st.caption(f"Pipeline: {result['retrieval']}")
    if result["error"]:
        # UI chỉ hiển thị thông báo an toàn và error code để tra server log.
        error = result["error"]
        # Session cũ có thể còn raw error string; thay bằng mã chung, không render lại.
        if not isinstance(error, dict):
            error = {
                "code": "RAG-SYS-100",
                "message": ERROR_CATALOG["RAG-SYS-100"],
            }
        st.error(f"{error['message']} Mã lỗi: `{error['code']}`")
    else:
        st.markdown(result["answer"])
    render_sources(result["sources"], key)


if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.title("🎮 GameDocs RAG Lab")
    st.caption("Phân tích patch notes, chính sách Riot, luật, bảo mật và thuật ngữ game.")
    st.divider()

    mode = st.radio(
        "Chế độ thử nghiệm",
        ["So sánh A/B", "Baseline RAG", "Advanced RAG"],
        help="A/B chạy cùng một câu hỏi qua hai cấu hình để so sánh trực tiếp.",
    )
    top_k = st.slider("Top-k evidence", 2, 10, 5)

    # Thu gọn các thiết lập kỹ thuật để sidebar dễ đọc; bấm vào tiêu đề để mở.
    with st.expander("⚙️ Advanced configuration", expanded=False):
        use_hyde = st.checkbox("HyDE query", value=True)
        use_bm25 = st.checkbox("BM25 lexical search", value=True, key="cfg_bm25")

        # RRF cần ít nhất hai ranked lists (Dense + BM25). Khi BM25 bị tắt,
        # đồng bộ state về False trước khi render để checkbox vừa khóa vừa unchecked.
        if "cfg_rrf" not in st.session_state:
            st.session_state.cfg_rrf = True
        if not use_bm25:
            st.session_state.cfg_rrf = False
        use_rrf = st.checkbox(
            "RRF fusion",
            key="cfg_rrf",
            disabled=not use_bm25,
        )
        use_pageindex = st.checkbox("PageIndex fallback", value=True)
        score_threshold = st.slider(
            "Cosine fallback threshold", 0.0, 1.0, 0.48, 0.01,
            disabled=not use_pageindex,
        )
        use_reorder = st.checkbox("Lost-in-the-middle reorder", value=True)

    advanced_config = {
        "use_hyde": use_hyde,
        "use_bm25": use_bm25,
        "use_rrf": use_rrf and use_bm25,
        "use_pageindex": use_pageindex,
        "score_threshold": score_threshold,
        "use_reorder": use_reorder,
    }
    baseline_config = {**BASELINE_CONFIG, "score_threshold": score_threshold}

    with st.expander("🧾 Tra cứu mã lỗi", expanded=False):
        for code, message in ERROR_CATALOG.items():
            st.markdown(f"`{code}` — {message}")

    st.divider()
    st.subheader("Câu hỏi gợi ý")
    suggestions = [
        "Tóm tắt các thay đổi lớn trong Patch 26.15.",
        "Riot thu thập và xử lý dữ liệu người chơi như thế nào?",
        "Giải thích Riot ID và Summoner Name khác nhau ra sao.",
        "Chính sách hoàn tiền nội dung trong game là gì?",
    ]
    for index, suggestion in enumerate(suggestions):
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_query = suggestion

    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.title("🎮 GameDocs RAG — Baseline vs Advanced")
st.caption(
    "Đặt cùng một câu hỏi để quan sát tác động của HyDE, BM25, RRF, "
    "PageIndex fallback và document reordering."
)

with st.expander("🧪 Cấu hình đang so sánh", expanded=False):
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="pipeline-card baseline"><b>Baseline RAG</b><br>'
                    '<span class="small-muted">Query → Dense cosine → LLM</span></div>',
                    unsafe_allow_html=True)
        st.json(baseline_config)
    with right:
        st.markdown('<div class="pipeline-card advanced"><b>Advanced RAG</b><br>'
                    '<span class="small-muted">HyDE → Dense + BM25 → RRF → '
                    'PageIndex fallback → Reorder → LLM</span></div>', unsafe_allow_html=True)
        st.json(advanced_config)

for turn_index, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(turn["query"])
    if turn["mode"] == "So sánh A/B":
        left, right = st.columns(2, gap="large")
        with left:
            render_result("Baseline RAG", turn["baseline"], "⚪", f"b_{turn_index}")
        with right:
            render_result("Advanced RAG", turn["advanced"], "🔴", f"a_{turn_index}")
    else:
        with st.chat_message("assistant"):
            render_result(turn["mode"], turn["result"], "🔴", f"r_{turn_index}")

typed_query = st.chat_input("Hỏi về patch notes, chính sách, luật hoặc thuật ngữ game...")
query = typed_query or st.session_state.pending_query
if query:
    st.session_state.pending_query = None
    with st.spinner("Đang chạy retrieval và generation..."):
        if mode == "So sánh A/B":
            baseline_result = run_pipeline(query, top_k, baseline_config)
            advanced_result = run_pipeline(query, top_k, advanced_config)
            st.session_state.history.append(
                {
                    "query": query,
                    "mode": mode,
                    "baseline": baseline_result,
                    "advanced": advanced_result,
                }
            )
        else:
            selected_config = baseline_config if mode == "Baseline RAG" else advanced_config
            st.session_state.history.append(
                {
                    "query": query,
                    "mode": mode,
                    "result": run_pipeline(query, top_k, selected_config),
                }
            )
    st.rerun()
