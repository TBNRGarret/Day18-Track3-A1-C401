# Individual Reflection — Lab 18

**(Nếu đề yêu cầu)** copy nội dung vào file `reflection_<HọTen>.md` và đổi `[Họ tên]` phía dưới cho đúng.

**Tên:** Vũ Hồng Quang - 2A202600341  
**Module phụ trách:** M2 — Hybrid Search (BM25 + Dense + RRF)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: **`src/m2_search.py`** — pipeline tìm kiếm hybrid cho RAG lab.
- Các hàm/class chính đã viết:
  - `segment_vietnamese()` — tách tiếng Việt (`underthesea.word_tokenize` khi có, fallback không phá flow).
  - `BM25Search` — index token + **`BM25Okapi`**, `search()` trả `SearchResult` với `method="bm25"`.
  - `DenseSearch` — embed (**`BAAI/bge-m3`**, `sentence-transformers`), index/upsert lên **Qdrant**, `search()` vector similarity (`method="dense"`).
  - `reciprocal_rank_fusion()` — gộp hai ranking (**RRF**, `k≈60`), `method="hybrid"`.
  - `HybridSearch` — gọi BM25 + Dense rồi RRF cho top-k cuối.
  - File test bổ sung (tùy nhóm): tích hép Qdrant thật trong `tests/test_m2_dense_qdrant_integration.py`.
- Số tests pass: **`tests/test_m2.py`** — **5/5**.

## 2. Kiến thức học được

- Khái niệm mới nhất:
  - **Hybrid retrieval**: lexical (BM25) bắt trùng từ cụ thể; **dense** bắt gần nghĩa khi cách diễn đạt khác nhau — kết hợp hai nguồn bằng **RRF** thay vì tin một thang đo duy nhất.
  - **Qdrant** như vector DB: lưu embedding + payload (metadata + text đầy đủ), tra cứu ANN có scale tốt hơn embed toàn corpus trong RAM mỗi lần.
- Điều bất ngờ nhất: RRF chỉ dùng **thứ hạng** (rank), không cần chuẩn hóa score BM25 và cosine-score dense — vẫn gộp ổn định.
- Kết nối với bài giảng (slide nào): phần **RAG retrieval** cổ điển vs production (chunk → retrieve top-k → generate), và **evaluation** downstream (Faithfulness/Relevancy phụ thuộc context lấy được).

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất:
  - BM25 nhạy với tokenization tiếng Việt (“nghỉ phép” một cụm vs hai token).
  - Đặt **`EMBEDDING_DIM`** và collection Qdrant khớp với `bge-m3`; nhớ **`recreate_collection`** khi đổi kích thước vector.
- Cách giải quyết: dùng `underthesea` khi cài được; tokenizer + lowercase + làm sạch dấu câu nhẹ quanh token cho BM25; đọc kỹ `config.py`; test Qdrant tách collection tạm để không ghi đè dữ liệu lab.
- Thời gian debug: (ước lượng) **~45–90 phút** tùy môi trường (lần đầu tải model, Qdrant chưa listen, sai port/docker).

## 4. Nếu làm lại

- Sẽ làm khác điều gì:
  - Thử **dense-only vs BM25-only vs hybrid** trên cùng test set nhỏ trước khi ráp pipeline — thấy nhanh module nào “win” dataset policy tiếng Việt của lab.
  - Thêm một vài assertion **HybridSearch** trong test (mock dense hoặc integration có flag) để regress khi TV5 chỉnh `pipeline`.
- Module nào muốn thử tiếp:
  - **M3 Reranking** (cross-encoder) để tinh chỉnh top-(20)→3 sau hybrid, hoặc **M4** để đọc Failure Analysis và gắn với sai lầm retrieval.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|----------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
