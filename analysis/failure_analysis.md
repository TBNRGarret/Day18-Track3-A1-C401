# Failure Analysis — Lab 18

**Nhóm:** Nhóm 1 (Ví dụ)
**Thành viên:** [Tên TV1 → M1] · [Tên TV2 → M5] · [Tên TV3 → M2] · [Tên BẠN → M4] · [Tên TV5 → M3]

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6500 | 0.8800 | +0.2300 |
| Answer Relevancy | 0.7000 | 0.8500 | +0.1500 |
| Context Precision | 0.5500 | 0.8000 | +0.2500 |
| Context Recall | 0.6000 | 0.8200 | +0.2200 |

## Bottom-5 Failures (Bắt bệnh hệ thống)

### #1
- **Question:** Làm thế nào để đăng ký bảo hiểm y tế cho nhân viên thử việc?
- **Expected:** Nhân viên thử việc chưa được tham gia bảo hiểm y tế theo quy định, phải đợi ký hợp đồng chính thức.
- **Got:** Bạn có thể đăng ký bảo hiểm y tế qua cổng thông tin nhân sự.
- **Worst metric:** Faithfulness (0.2)
- **Error Tree:** Output sai → Context đúng (có nói về việc thử việc không có BHYT) → Query OK → Root cause: LLM Hallucination (Tự bịa câu trả lời thay vì đọc context).
- **Suggested fix:** Thêm chỉ thị mạnh vào Prompt (M3/Pipeline): "Tuyệt đối không tự bịa câu trả lời. Nếu context không hỗ trợ, hãy nói Không biết".

### #2
- **Question:** Quy trình xin nghỉ phép quá 5 ngày liên tục?
- **Expected:** Cần sự phê duyệt của Giám đốc khối và nộp đơn trước 2 tuần.
- **Got:** Tôi không tìm thấy thông tin về quy trình nghỉ phép.
- **Worst metric:** Context Recall (0.0)
- **Error Tree:** Output sai → Context SAI (Retrieval lấy nhầm quy trình nghỉ ốm) → Query OK → Root cause: Retrieval fail (M2). Từ khóa "nghỉ phép" bị loãng.
- **Suggested fix:** Tăng trọng số của BM25 (Keyword search) trong Hybrid Search (M2) để bắt dính cụm từ "nghỉ phép quá 5 ngày".

### #3
- **Question:** Trợ cấp ăn trưa là bao nhiêu tiền một ngày?
- **Expected:** 50,000 VND / ngày làm việc thực tế.
- **Got:** Công ty có cung cấp trợ cấp ăn trưa cho nhân viên.
- **Worst metric:** Answer Relevancy (0.4)
- **Error Tree:** Output thiếu thông tin → Context đúng → Query OK → Root cause: LLM Generation quá ngắn gọn.
- **Suggested fix:** Yêu cầu LLM trong Prompt trích xuất con số cụ thể nếu câu hỏi có chứa từ "bao nhiêu tiền", "số lượng".

### #4
- **Question:** Bao giờ thì nhận được thưởng tháng 13?
- **Expected:** Kỳ lương tháng 1 năm sau.
- **Got:** Không có thông tin.
- **Worst metric:** Context Precision (0.3)
- **Error Tree:** Output sai → Context bị trôi (Chunk chứa đáp án nằm ở vị trí thứ 9) → Root cause: Reranking (M3) không đẩy được chunk quan trọng lên top.
- **Suggested fix:** Đổi mô hình Cross-Encoder (M3) sang bản tiếng Việt (vd: bge-reranker-v2-m3) để Rerank tốt hơn.

### #5
- **Question:** Khách hàng phàn nàn về thái độ tài xế thì gọi số nào?
- **Expected:** Gọi hotline CSKH 1900 xxxx.
- **Got:** Gọi hotline 1900 xxxx.
- **Worst metric:** Faithfulness (0.7 - thấp nhất trong các câu đúng)
- **Error Tree:** Output đúng nhưng cụt lủn → Context đúng → Query OK → Root cause: Prompt chưa định hướng tone giọng trả lời.
- **Suggested fix:** Bổ sung System Prompt: "Trả lời lịch sự, đầy đủ chủ vị".

---

## Case Study (Dành cho thuyết trình)

**Question (Ví dụ #2):** Quy trình xin nghỉ phép quá 5 ngày liên tục?

**Error Tree walkthrough:**
1. Output đúng? → **Không** (AI bảo không biết).
2. Context đúng? → **Không** (Hệ thống tìm kiếm trả về nhầm quy trình "nghỉ ốm" do có nhiều từ khóa trùng lặp).
3. Query rewrite OK? → **Có**.
4. Fix ở bước: **M2 (Retrieval)**.

**Phân tích sâu:** Lỗi này rất phổ biến ở các văn bản hành chính nhân sự. Vector search chỉ thấy các chunk có nghĩa "nghỉ ngơi" giống nhau nên lấy bừa. 
**Nếu có thêm 1 giờ:** Nhóm sẽ tăng trọng số BM25 (Keyword search) trong Hybrid Search để "Neo" cứng cụm từ "nghỉ phép quá 5 ngày", giúp Retrieval chính xác 100%.
