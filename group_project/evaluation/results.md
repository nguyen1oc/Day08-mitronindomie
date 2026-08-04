# RAG Evaluation Results

## Framework sử dụng

> Framework: RAGAS / Heuristic Evaluation Framework

---

## Overall Scores

| Metric | Config A (Hybrid + RRF Rerank) | Config B (Dense-Only) | Δ (Difference) |
|--------|-------------------------------|----------------------|----------------|
| Faithfulness | 0.926 | 0.560 | +0.366 |
| Answer Relevance | 0.884 | 0.510 | +0.374 |
| Context Recall | 0.890 | 0.460 | +0.430 |
| Context Precision | 0.746 | 0.410 | +0.336 |
| **Average** | **0.862** | **0.485** | **+0.377** |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + RRF Rerank):**
> Kết hợp Sparse Search (BM25) và Dense Search (Semantic Search BGE-M3) thông qua thuật toán Reciprocal Rank Fusion (RRF k=60). Có tích hợp bẫy Fallback cho câu hỏi out-of-domain.

**Config B (Dense-Only):**
> Chỉ sử dụng Vector Search dựa trên Cosine Similarity, không dùng BM25 và không qua RRF Reranking.

**Kết luận:**
> Config A (Hybrid + RRF) cho điểm số vượt trội hơn rõ rệt so với Config B (Dense-Only) trên tất cả 4 tiêu chí. Việc kết hợp BM25 giúp giải quyết triệt để các câu hỏi chứa từ khóa chính xác (số bản vá, tên tướng, mã trang phục) mà Vector Search đơn thuần dễ bị bỏ sót.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Thử nghiệm câu hỏi phức tạp | 0.82 | 0.80 | 0.85 | Retrieval | Từ khóa hiếm |

---

## Recommendations

### Cải tiến 1: Tối ưu Tokenizer Tiếng Việt cho BM25
**Action:** Tích hợp `underthesea` hoặc `pyvi` để ghép các từ ghép tiếng Việt trước khi đưa vào BM25.
**Expected impact:** Tăng chỉ số Context Precision lên thêm 5-8%.

### Cải tiến 2: Tinh chỉnh Ngưỡng Fallback Score Threshold
**Action:** Điều chỉnh ngưỡng Cosine Similarity Threshold ở Task 9 từ 0.48 thành 0.45 sau khi thử nghiệm bộ câu hỏi thực tế.
**Expected impact:** Loại bỏ hoàn toàn các trường hợp trả lời rác cho câu hỏi out-of-domain.

### Cải tiến 3: Áp dụng Cross-Encoder Reranking
**Action:** Bổ sung mô hình Jina Reranker v2 làm bước Reranker thứ 2 sau khi qua RRF.
**Expected impact:** Tăng chỉ số Answer Relevance và hạn chế tối đa hiện tượng Lost in the Middle.
