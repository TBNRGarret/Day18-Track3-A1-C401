# Cá nhân: Vũ Lê Hoàng
**Vai trò trong nhóm:** Thành viên 4 (Quality Analyst)
**Module phụ trách:** M4 (Evaluation) & Failure Analysis

---

## 1. Công việc đã thực hiện
Trong bài Lab này, em đảm nhận việc viết code cho **Module 4 (Evaluation)** và phân tích lỗi toàn hệ thống:
- Sử dụng framework **RAGAS** để tự động chấm điểm hệ thống qua 4 metrics cốt lõi: *Faithfulness, Answer Relevancy, Context Precision, Context Recall*.
- Xử lý các lỗi dữ liệu (NaN values) trả về từ RAGAS để tính toán điểm số trung bình một cách chính xác.
- Thu thập top 5 câu hỏi có điểm thấp nhất (Bottom-5) và áp dụng mô hình **Error Tree (Cây chẩn đoán)** để truy ngược từ Output -> Context -> Query nhằm tìm ra root cause của lỗi (do Retrieval hay do Generation).
- Soạn thảo file `failure_analysis.md` cung cấp insight trực tiếp cho nhóm.

## 2. Khó khăn gặp phải
- Framework RAGAS thi thoảng trả về các giá trị `NaN` do lỗi trong quá trình LLM chấm điểm, khiến cho việc tính `avg_score` bị sập. Em đã phải viết thêm logic fallback để handle các giá trị này (đưa về 0.0) nhằm đảm bảo pipeline không bị gián đoạn.
- Khi phân tích Error Tree, đôi lúc rất khó phân định rạch ròi giữa việc "Context sai do Chunking" hay "Context sai do Search", đòi hỏi phải in trực tiếp metadata của chunk ra để đọc và suy luận.

## 3. Bài học rút ra
- Cảm nhận rõ ràng nhất là: **"Điểm số không nói lên tất cả"**. Một pipeline đạt điểm 0.8 chưa chắc đã ngon nếu chúng ta không hiểu tại sao 0.2 còn lại bị sai. 
- Failure Analysis (bắt bệnh) quan trọng hơn việc mù quáng thay thế model. Việc chỉ ra đúng hệ thống đang bị *Hallucination* (lỗi M3/Prompt) hay *Missing relevant chunks* (lỗi M1/M2) giúp nhóm tiết kiệm rất nhiều thời gian thay vì "thử và sai" vô định.

## 4. Hướng cải thiện (Nếu có thêm 1 giờ)
- Em sẽ viết thêm một số Custom Metrics (ví dụ: Toxicity hoặc Tone/Style checker) để đánh giá xem câu trả lời của AI có chuẩn giọng điệu doanh nghiệp không, ngoài việc chỉ đúng thông tin.
- Tạo thêm một bộ Test Set (Edge cases) với các câu hỏi bẫy hoặc câu hỏi cần suy luận tổng hợp từ nhiều nguồn tài liệu để stress-test hệ thống.
