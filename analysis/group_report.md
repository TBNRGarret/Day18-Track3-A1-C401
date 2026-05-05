# Báo cáo nhóm — Lab 18: Production RAG

**Khóa / lớp:** AICB-P2T3 · Track 3 · **Nhóm:** A1–C401  
**Ngày hoàn thành:** 04/05/2026  

---

## 1. Giới thiệu

Nhóm triển khai một pipeline RAG kiểu **production**: chunking nâng cao (M1) → enrichment trước index (M5) → **hybrid retrieval** BM25 + dense trên **Qdrant** (M2) → rerank Cross-Encoder (M3) → sinh câu trả lời có ràng buộc context → đánh giá **RAGAS** (M4). Hệ thống được đối chiếu với **naive baseline** (chunk đoạn văn + dense-only) trong cùng tập đánh giá.

---

## 2. Thành viên & phân công

| Thành viên | MSSV | Module chính | Ghi chú công việc | Auto-test (pytest) module |
|------------|------|--------------|-------------------|---------------------------|
| **Đàm Lê Văn Toàn** | 2A202600017 | M1 Chunking | Hierarchical / semantic / structure-aware chunking | `tests/test_m1.py` — **13/13** |
| **Vũ Hồng Quang** | 2A202600341 | M2 Hybrid Search | BM25 (UndertheSea) + Dense bge-m3 + Qdrant + RRF | `tests/test_m2.py` — **5/5** |
| **Đặng Tiến Dũng** | 2A202600024 | M3 + tích hợp pipeline | Cross-Encoder rerank, benchmark; ghép flow trong `pipeline.py` | `tests/test_m3.py` — **5/5** |
| **Vũ Lê Hoàng** | 2A202600342 | M4 Evaluation | RAGAS (`evaluate_ragas`), báo lỗi / failure diagnostics | `tests/test_m4.py` — **4 passed**, 1 skip (smoke integration) |
| **Nguyễn Quang Trường** | — | M5 Enrichment | HyQA, contextual prepend, metadata extraction, pipeline enrich | `tests/test_m5.py` — **10/10** |

*Bổ sung:* file tích hợp **Qdrant thật** `tests/test_m2_dense_qdrant_integration.py` (marker **integration**) — chỉ khi có Qdrant local.

Chi tiết Bottom-5 + Error Tree: **`analysis/failure_analysis.md`** (Người điều phối minh chứng điểm B3 theo Rubric.)

---

## 3. Kiến trúc pipeline (Production)

Luồng chạy chính (`src/pipeline.py` / `python main.py`):

1. **M1:** `load_documents` → `chunk_hierarchical` → danh sách child kèm `parent_id`/metadata.
2. **M5:** `enrich_chunks` — contextual prepend, HyQA, metadata — dùng `enriched_text` khi enrichment thành công (fallback chunks gốc nếu không).
3. **M2:** `HybridSearch.index` — BM25 in-memory + vector index trên **Qdrant**.
4. **M3:** `CrossEncoderReranker.rerank` giới hạn ngữ cảnh đưa vào generator.
5. **Generate:** prompt “chỉ dùng context”, fallback khi thiếu API key trong lab.
6. **M4:** RAGAS 4 metric + báo cáo JSON.

---

## 4. Kết quả định lượng (RAGAS)

*Nguồn:* `reports/naive_baseline_report.json` và `reports/ragas_report.json` (cùng tập **`num_questions`** ghi trong report.)

| Metric | Naive baseline | Production | Δ |
|--------|----------------|------------|---|
| **Faithfulness** | 0.65 | **0.88** | **+0.23** |
| **Answer relevancy** | 0.70 | **0.85** | **+0.15** |
| **Context precision** | 0.55 | **0.80** | **+0.25** |
| **Context recall** | 0.60 | **0.82** | **+0.22** |

**Nhận xét Rubric · B2:** Cả **bốn** metric đều ≥ **0,75**, đạt tiêu chí nhóm mạnh. **Bonus Faithfulness:** 0.88 ≥ 0.85 → đạt nhánh **faithfulness**.

**Điều kiện dữ liệu:** báo cáo hiện trên **`num_questions`: 3** trong JSON; nhóm khuyến nghị mở rộng **`test_set.json`** đủ quy mô (~20+) để điểm ổn định hơn và phản ánh tốt hơn các lỗi edge-case (theo Reflection M4.)

---

## 5. Nguyên nhân cải thiện chính (so với Naive)

- **Recall & lexical:** Hybrid **BM25 + dense** và **RRF** giảm rủi ro “dense trượt cụm từ” so với dense-only trong baseline — hỗ trợ **context recall** và **precision**.
- **Ngữ nghĩa & chủ đề:** Chunk **parent–child** (M1) + enrichment (M5) giúp đoạn index mang thêm ngữ cảnh / cầu nối từ vựng (HyQA).
- **Chất lượng ngữ cảnh đầu vào LLM:** Rerank (M3) đẩy chunk liên quan lên đầu → LLM ít phải ghép ý giữa nhiều đoạn nhiễu → **faithfulness**.
- **Prompt có ràng buộc:** Hướng dẫn “chỉ dùng context / không đủ thì báo không tìm thấy” góp phần hạ hallucination trong phạm vi lab.

---

## 6. Failure analysis / rủi ro còn lại

- Phân tích **Bottom-5** + **walkthrough Error Tree**: xem **`analysis/failure_analysis.md`** (đủ các case minh chứng Retrieval vs Generation hay Prompt).
- Khi enrichment hoặc API LLM không ổn định, pipeline có nhánh fallback — cần theo dõi log để phân biệt “điểm thấp do eval” và “điểm thấp do hạ tầng”.

---

## 7. Gợi ý cho thuyết trình (~5')

1. **Số:** Bảng RAGAS naive vs production (mục 4) + 1 câu về **hybrid + rerank**.
2. **Win lớn nhất:** M2 hybrid + M3 rerank (dễ minh họa bằng 1 câu hỏi khó với từ khóa hiếm).
3. **Case study:** Chọn **1 failure** trong `failure_analysis.md`, đi **Error Tree** Output → Context → Query.
4. **Nếu thêm 1 giờ:** mở rộng test set, tinh chỉnh trọng số RRF/BM25 theo domain, hoặc reranker tiếng Việt nếu domain HR policy.

---

## 8. Kết luận

Pipeline production chạy end-to-end, có metric RAGAS vượt ngưỡng chấm nhóm, có tài liệu phân tích lỗi chi tiết. Hướng tiếp theo: **tăng kích thước & đa dạng test set**, theo dõi **latency** (breakdown đã gắn trong pipeline khi chạy `evaluate_pipeline`) và canh chỉnh chi phí API khi scale.
