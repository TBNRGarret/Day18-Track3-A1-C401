# KẾ HOẠCH THỰC HIỆN BÀI TẬP NHÓM: PRODUCTION RAG SYSTEM

## 1. Phân công nhiệm vụ (5 thành viên)

### **Thành viên 1: Data Architect (M1 + Presentation)**
*   **Module chính:** **M1 (Chunking)**. Đảm bảo dữ liệu được xử lý, làm sạch và cắt nhỏ tối ưu.
*   **Nhiệm vụ phụ:** Trực tiếp soạn Slide và chuẩn bị nội dung thuyết trình (Presentation - 10đ) để bù đắp cho phần code M1 khá nhẹ.

### **Thành viên 2: AI Specialist (M5)**
*   **Module chính:** **M5 (Enrichment)**. Tập trung cao độ code 4 hàm LLM (Summarization, HyQA, Contextual Prepend, Auto Metadata) để lấy điểm Bonus (+3đ).

### **Thành viên 3: Retrieval Specialist (M2 + Reporting)**
*   **Module chính:** **M2 (Search)**. Xây dựng hệ thống tìm kiếm Hybrid (BM25 + Vector).
*   **Nhiệm vụ phụ:** Viết báo cáo tổng kết nhóm `analysis/group_report.md`.

### **Thành viên 4: Quality & Insights (M4 + Analysis)**
*   **Module chính:** **M4 (Evaluation)**. Chạy RAGAS evaluation để chấm điểm hệ thống.
*   **Nhiệm vụ phụ:** Chịu trách nhiệm toàn bộ phần Failure Analysis (10đ) — Bắt bệnh hệ thống qua Error Tree cho 5 câu điểm thấp nhất.

### **Thành viên 5: System Integrator (M3 + Pipeline + Bonus)**
*   **Module chính:** **M3 (Reranking)**.
*   **Nhiệm vụ phụ:** Ráp nối `src/pipeline.py`. Đo lường **Latency breakdown (+2đ)**. Tinh chỉnh LLM Prompt ở bước Generate để kéo điểm Faithfulness **>= 0.85 (+5đ)**.

---

## 2. Quy trình phối hợp
1. **Giai đoạn 1 (Code độc lập):** TV1, TV2, TV3 làm M1, M5, M2. TV5 làm trước khung `pipeline.py`.
2. **Giai đoạn 2 (Tích hợp & Chấm điểm):** TV5 gom code M1,M2,M3,M5 vào chạy thử. TV4 lấy kết quả chạy `ragas_report.json`.
3. **Giai đoạn 3 (Báo cáo & Hoàn thiện):** TV4 làm Failure Analysis. TV3 viết Báo cáo. TV1 làm Slide. TV5 chạy `check_lab.py` để nộp bài.
