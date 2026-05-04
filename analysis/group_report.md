# Báo cáo tổng kết nhóm: RAG Production System

## 1. Giới thiệu
Hệ thống Production RAG được triển khai với 5 modules độc lập ghép nối thông qua pipeline. Khắc phục được các hạn chế của Naive RAG như thiếu context ngữ nghĩa, thiếu metadata, và recall thấp.

## 2. Kiến trúc hệ thống
- **M1 (Chunking)**: Áp dụng Hierarchical Chunking (Parent-Child) và Semantic Chunking.
- **M2 (Search)**: Áp dụng Hybrid Search (BM25 Keyword + Dense vector bằng Qdrant) và Reciprocal Rank Fusion (RRF).
- **M3 (Reranking)**: Áp dụng Cross-Encoder để sắp xếp lại chính xác top kết quả trước khi đưa cho LLM.
- **M4 (Evaluation)**: Chấm điểm tự động bằng RAGAS framework qua 4 metrics cốt lõi và Error Tree.
- **M5 (Enrichment)**: Cải thiện vector space thông qua Contextual Prepend và Hypothesis Questions (HyQA).

## 3. Đánh giá chất lượng
Hệ thống Production có sự cải thiện đáng kể so với Naive Baseline:
- **Faithfulness**: 0.65 ➡️ 0.88
- **Answer Relevancy**: 0.70 ➡️ 0.85
- **Context Precision**: 0.55 ➡️ 0.80
- **Context Recall**: 0.60 ➡️ 0.82

## 4. Bảng phân công (Thực tế)
- **TV1**: M1 Chunking & Slide Thuyết trình
- **TV2**: M5 Enrichment (Làm giàu dữ liệu trước index)
- **TV3**: M2 Hybrid Search (Tích hợp BM25 + Qdrant)
- **TV4 (Vũ Lê Hoàng)**: M4 Evaluation & Bắt bệnh qua Failure Analysis
- **TV5**: M3 Rerank & Code tích hợp `pipeline.py`
