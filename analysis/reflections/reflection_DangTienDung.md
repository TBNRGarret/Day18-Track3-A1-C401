# Individual Reflection — Lab 18: Production RAG

**Tên:** Dang Tien Dung
**MSSV:** 2A202600024  
**Module phụ trách:** M3 (Reranking)
**Vai trò:** System Integrator (M3 + Pipeline + Bonus)

---

## 1. Đóng góp kỹ thuật cụ thể

### 1.1. M3 — Reranking (Cross-Encoder)

- Tích hợp reranker cross-encoder cho bước post-retrieval nhằm cải thiện **context precision** và gián tiếp tăng **faithfulness**.
- Implement `CrossEncoderReranker` trong `src/m3_rerank.py`:
  - `_load_model()` hỗ trợ **fallback**: ưu tiên `FlagEmbedding.FlagReranker`, nếu không có thì dùng `sentence_transformers.CrossEncoder`.
  - `rerank()` tạo pairs `(query, doc_text)`, lấy điểm (compute/predict), sort giảm dần và trả về top-k `RerankResult`.
  - `benchmark_reranker()` đo latency trung bình/min/max theo số lần chạy.

### 1.2. (Optional) FlashrankReranker

- Implement `FlashrankReranker` trong `src/m3_rerank.py`:
  - Load `flashrank.Ranker` lazy.
  - Chạy rerank với `RerankRequest(query, passages)`.
  - Có cơ chế fallback (giữ thứ tự ban đầu) nếu environment thiếu dependency hoặc API thay đổi.

### 1.3. Ráp nối pipeline nhóm + đo latency breakdown

- Cập nhật `src/pipeline.py` để pipeline chạy theo flow:
  - Chunking (M1) → Enrichment (M5) → Indexing/Search (M2) → Rerank (M3) → Generate Answer → Evaluate (M4)
- Thêm đo thời gian (ms) cho các bước chính:
  - Build-time: `chunking_ms`, `enrichment_ms`, `indexing_ms`, `reranker_init_ms`
  - Query-time: `search_ms`, `rerank_ms`, `generate_ms`
- Ghi latency summary vào report (`ragas_report.json`) qua trường `latency_ms` để phục vụ bonus **Latency breakdown (+2đ)**.

### 1.4. Gợi ý hướng tối ưu prompt generation để kéo Faithfulness

- Đề xuất hướng prompt để tăng **faithfulness** cho bước Generate (không tự nhận đã hoàn thiện phần này):
  - ràng buộc **chỉ dùng context**, thiếu evidence thì trả về `"Không tìm thấy thông tin."`
  - format context theo tag `[C1]...[Ck]` để dễ trích dẫn
- Lưu ý: phần LLM generation/prompt tuning cần được các thành viên thống nhất và chạy end-to-end để kiểm chứng bằng RAGAS.

---

## 2. Kiến thức học được (liên hệ bài giảng)

- Hiểu rõ vai trò của **reranking** trong hệ thống RAG production:
  - Retrieval (BM25/dense/hybrid) tối ưu recall, nhưng rerank giúp tăng precision ở top-k.
- Thấy sự khác biệt giữa:
  - **Bi-encoder** (embedding search) nhanh nhưng coarse.
  - **Cross-encoder reranker** chậm hơn nhưng “deep matching” tốt hơn.
- Nhận ra faithfulness không chỉ đến từ retrieval mà còn phụ thuộc mạnh vào:
  - prompt constraints
  - refusal behavior khi thiếu evidence
  - deterministic decoding (temperature thấp)

---

## 3. Khó khăn gặp phải & cách giải quyết

- **Dependency / environment issues (pytest, dotenv, model packages):**
  - Một số package tải chậm/timeout, gây lỗi import (`dotenv`, `pytest`).
  - Giải pháp: cài các dependency tối thiểu trước để unblock test/pipeline; trong code thì thêm fallback để hệ thống vẫn chạy khi thiếu optional packages.

- **Interface khác nhau giữa các thư viện reranker:**
  - `FlagReranker.compute_score()` vs `CrossEncoder.predict()`
  - Giải pháp: detect interface bằng `hasattr()` và thống nhất output về `RerankResult`.

- **Đảm bảo pipeline không “hallucinate”:**
  - Giải pháp: prompt “ONLY context” + câu từ chối cứng khi thiếu bằng chứng.

---

## 4. Tự đánh giá

**Mức độ đóng góp (1–5):** 5/5

- Hoàn thành module M3 (reranking) và tích hợp pipeline.
- Thực hiện latency breakdown cho bonus.
- Đề xuất hướng prompt để tăng faithfulness và đảm bảo pipeline chạy ổn định khi thiếu dependency.
