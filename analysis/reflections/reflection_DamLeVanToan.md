# Individual Reflection — Lab 18

**Tên:** Đàm Lê Văn Toàn  
**MSSV:** 2A202600017  
**Module phụ trách:** M1 (Chunking)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: **M1 — Advanced Chunking Strategies**
- Các hàm/class chính đã viết:
  - `chunk_semantic()` — Phân đoạn văn bản theo độ tương đồng ngữ nghĩa giữa các câu bằng mô hình `BAAI/bge-m3`, nhóm các câu cùng chủ đề vào cùng một chunk dựa trên ngưỡng cosine similarity.
  - `chunk_hierarchical()` — Tạo cấu trúc parent-child: chia văn bản thành parent chunk (ngữ cảnh rộng) và child chunk (đơn vị nhỏ để embedding chính xác), mỗi child liên kết với parent qua `parent_id`.
  - `chunk_structure_aware()` — Phân tích cấu trúc Markdown (header `#`, `##`, `###`) để tách chunk theo từng section logic, giữ nguyên bảng, danh sách, code block.
  - `compare_strategies()` — Chạy cả 4 chiến lược trên toàn bộ tài liệu và in bảng so sánh (số chunk, độ dài trung bình, min, max).
- Số tests pass: **13/13 (100%)**

## 2. Kiến thức học được

- Khái niệm mới nhất: **Hierarchical Chunking** (parent-child pattern) — chiến lược production-ready trong RAG: index child vào vector DB để embedding chính xác, nhưng khi trả lời LLM thì dùng parent để có đủ ngữ cảnh.
- Điều bất ngờ nhất: Cosine similarity của `BAAI/bge-m3` với tiếng Việt cao hơn nhiều so với `all-MiniLM-L6-v2`, giúp phân đoạn semantic chính xác hơn mà không cần điều chỉnh threshold thủ công.
- Kết nối với bài giảng: Slide "Chunking Strategies" — phần so sánh basic vs. semantic vs. hierarchical, và lý do tại sao không nên chunk cố định theo số ký tự.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Xử lý đúng edge case trong `chunk_hierarchical()` — trường hợp văn bản ngắn hơn `child_size`, hoặc paragraph cuối cùng không đủ kích thước `parent_size` nhưng vẫn phải được gom vào parent.
- Cách giải quyết: Thêm điều kiện flush buffer sau vòng lặp (`if current_text.strip()`) cho cả parent và child; dùng sliding window đơn giản để chia child từ parent text.
- Thời gian debug: ~30 phút cho test `test_hierarchical_valid_parent_ids` — do `parent_id` trong metadata của parent chunk ban đầu bị đặt sai key.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Thêm overlap giữa các child chunk (ví dụ 50 ký tự) để không mất ngữ cảnh ở ranh giới chunk — hiện tại sliding window cắt cứng mà không overlap.
- Module nào muốn thử tiếp: **M5 (Enrichment)** — phần sinh HyQA (Hypothetical Questions & Answers) để làm phong phú metadata chunk trước khi đưa vào vector DB rất thú vị.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 4 |
