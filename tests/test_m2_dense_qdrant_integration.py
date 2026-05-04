"""
Tích hợp DenseSearch ↔ Qdrant (test thật trên vector DB).

Qdrant làm gì?
  - Đây là **vector database**: lưu embedding (vector số chiều cao) của từng chunk tài liệu
    cùng **payload** (metadata + full text để hiển thị).
  - Tra cứu: embed câu hỏi → Qdrant tìm vector **gần nhất** theo cosine (similarity semantic),
    không cần quét hết corpus như BM25 trong RAM.
  - Trong lab: `DenseSearch.index()` recreate collection + upsert points;
    `DenseSearch.search()` gửi query vector → Qdrant trả top-k có điểm cao nhất.

Chạy: `pytest tests/test_m2_dense_qdrant_integration.py -m integration`

Yêu cầu môi trường: Qdrant đang listen (vd. docker compose up), đã pip install qdrant-client,
sentence-transformers; lần đầu có thể tải BAAI/bge-m3.
"""
from __future__ import annotations

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import QDRANT_HOST, QDRANT_PORT  # noqa: E402
from src.m2_search import DenseSearch  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture
def ephemeral_dense_collection():
    """Tạo collection tên ngẫu nhiên, xóa sau test để không đụng lab18_production."""
    name = f"lab18_pytest_dense_{uuid.uuid4().hex[:12]}"
    yield name
    try:
        from qdrant_client import QdrantClient

        c = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
        c.delete_collection(collection_name=name)
    except Exception:
        pass


CHUNKS = [
    {"text": "Nhân viên được nghỉ phép năm 12 ngày theo quy định nội bộ.", "metadata": {"topic": "policy"}},
    {"text": "Mật khẩu phải đổi tối thiểu mỗi 90 ngày.", "metadata": {"topic": "it"}},
    {"text": "Thử việc kéo dài tối đa 60 ngày làm việc.", "metadata": {"topic": "hr"}},
]


@pytest.mark.skipif(not _qdrant_reachable(), reason="Không kết nối được Qdrant (bật server tại localhost:6333?).")
def test_dense_index_persists_points_in_qdrant(ephemeral_dense_collection):
    from qdrant_client import QdrantClient

    coll = ephemeral_dense_collection
    search = DenseSearch()
    search.index(CHUNKS, collection=coll)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
    cnt = client.count(collection_name=coll, exact=True)
    assert cnt.count == len(CHUNKS)


@pytest.mark.skipif(not _qdrant_reachable(), reason="Không kết nối được Qdrant (bật server tại localhost:6333?).")
def test_dense_search_queries_qdrant_and_returns_scores(ephemeral_dense_collection):
    """Gọi client.search của Qdrant qua DenseSearch.search — semantic top-1 nên nhớ các chữ trong chunk nghỉ phép."""
    coll = ephemeral_dense_collection
    search = DenseSearch()
    search.index(CHUNKS, collection=coll)

    results = search.search("chính sách nghỉ phép nhân viên được mấy ngày một năm", top_k=2, collection=coll)

    assert len(results) >= 1
    assert all(r.method == "dense" for r in results)
    assert all(isinstance(r.score, float) and r.score > 0 for r in results)
    top = results[0].text
    low = top.lower()
    assert "nghỉ" in low and ("12" in top or "ngày" in low)
