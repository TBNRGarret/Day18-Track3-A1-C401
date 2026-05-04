# Individual Reflection — Lab 18

**Tên:** [Nguyễn Quang Trường]  
**Module phụ trách:** [M5]

---

## 1. Đóng góp kỹ thuật

- Module đã implement: **M5: RAG Enrichment Pipeline**
- Các hàm/class chính đã viết:
  - `summarize_chunk(text: str) -> str`: Tóm tắt chunk bằng GPT-4o-mini (có fallback).
  - `generate_hypothesis_questions(text: str, n_questions: int) -> list[str]`: Tạo câu hỏi giả thuyết (HyQA) để thu hẹp khoảng cách từ vựng.
  - `contextual_prepend(text: str, document_title: str) -> str`: Thêm context prepend từ Anthropic.
  - `extract_metadata(text: str) -> dict`: Trích xuất metadata tự động (topic, entities, category, language) và xử lý JSON format an toàn.
  - `enrich_chunks(chunks: list[dict], methods: list[str]) -> list[EnrichedChunk]`: Pipeline hoàn chỉnh để làm giàu hàng loạt chunk.
- Số tests pass: **10/10** (pytest tests/test_m5.py pass 100%)

## 2. Kiến thức học được

- **Khái niệm mới nhất**: Các kỹ thuật làm giàu dữ liệu trước khi embedding (Pre-retrieval optimization) như HyQA hay Contextual Prepend giúp tăng đáng kể chất lượng retrieval.
- **Điều bất ngờ nhất**: Chỉ bằng việc thêm 1 câu context mô tả vị trí của chunk (contextual prepend) hoặc các câu hỏi giả định, LLM đã giúp mô hình hiểu sâu hơn hẳn về semantic relevance.
- **Kết nối với bài giảng**: Kỹ thuật xử lý dữ liệu trước khi index (Data pipeline / Indexing optimization) và giải quyết vấn đề Vocabulary gap.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất**: Khi dùng API OpenAI để sinh JSON cho metadata, LLM thi thoảng sinh ra markdown formatting (vd: ` ```json ... ``` `) dẫn đến lỗi `JSONDecodeError`.
- **Cách giải quyết**: Xây dựng đoạn mã dùng thư viện Regex (`re`) để tự động nhận diện và bóc tách chuỗi JSON sạch trước khi gọi `json.loads()`.
- **Thời gian debug**: ~20 phút (phần lớn thời gian để test các prompt và fix regex JSON).

## 4. Nếu làm lại

- **Sẽ làm khác điều gì**: Áp dụng xử lý bất đồng bộ (async) cho các hàm gọi API OpenAI để tăng tốc quá trình làm giàu dữ liệu (enrichment process) thay vì chạy vòng lặp đồng bộ tuần tự như hiện tại.
- **Module nào muốn thử tiếp**: Module M2 (Hybrid Search) để xem dữ liệu sau khi được enrichment sẽ giúp BM25 và Dense Search hoạt động hiệu quả hơn như thế nào.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 5 |
| Problem solving | 5 |

**Tổng cộng: 20/20**
