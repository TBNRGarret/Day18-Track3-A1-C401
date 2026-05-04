"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Implement Vietnamese word segmentation
    # 1. from underthesea import word_tokenize
    # 2. return word_tokenize(text, format="text")
    # Why: BM25 needs word boundaries. "nghỉ phép" = 1 word, not 2.
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        pass
    return text  # fallback


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        # Implement BM25 indexing
        # 1. self.documents = chunks
        # 2. For each chunk: segment_vietnamese(chunk["text"]) → split by space
        # 3. self.corpus_tokens = [tokenized list for each chunk]
        # 4. from rank_bm25 import BM25Okapi
        #    self.bm25 = BM25Okapi(self.corpus_tokens)
        self.documents = chunks
        self.corpus_tokens = []
        punct = ".,;:!?\"'()"
        for chunk in chunks:
            seg = segment_vietnamese(chunk["text"]).lower().split()
            tokens = []
            for t in seg:
                t = t.strip(punct).strip('"').strip("'")
                if t:
                    tokens.append(t)
            self.corpus_tokens.append(tokens)
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        # Implement BM25 search
        # 1. tokenized_query = segment_vietnamese(query).split()
        # 2. scores = self.bm25.get_scores(tokenized_query)
        # 3. top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # 4. Return [SearchResult(text=..., score=..., metadata=..., method="bm25")]
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = segment_vietnamese(query).lower().split()
        punct = ".,;:!?\"'()"
        tokenized_query = [t.strip(punct).strip('"').strip("'") for t in tokenized_query if t.strip(punct)]
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for i in top_indices:
            doc = self.documents[i]
            out.append(SearchResult(
                text=doc["text"],
                score=float(scores[i]),
                metadata=dict(doc.get("metadata", {})),
                method="bm25",
            ))
        return out


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        # XONG: Implement dense indexing
        # 1. from qdrant_client.models import Distance, VectorParams, PointStruct
        # 2. self.client.recreate_collection(collection, VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        # 3. texts = [c["text"] for c in chunks]
        # 4. vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        # 5. points = [PointStruct(id=i, vector=v.tolist(), payload={**c["metadata"], "text": c["text"]}) ...]
        # 6. self.client.upsert(collection, points)
        from qdrant_client.models import Distance, VectorParams, PointStruct
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = []
        for i, (v, c) in enumerate(zip(vectors, chunks)):
            md = dict(c.get("metadata", {}))
            vec = v.tolist() if hasattr(v, "tolist") else list(v)
            payload = {**md, "text": c["text"]}
            points.append(PointStruct(id=i, vector=vec, payload=payload))
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # XONG: Implement dense search
        # 1. query_vector = self._get_encoder().encode(query).tolist()
        # 2. resp = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k); hits = resp.points
        # 3. Return [SearchResult(text=hit.payload["text"], score=hit.score, metadata=hit.payload, method="dense")]
        qv = self._get_encoder().encode(query)
        query_vector = qv.tolist() if hasattr(qv, "tolist") else list(qv)
        resp = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
        )
        hits = resp.points
        out: list[SearchResult] = []
        for hit in hits:
            pl = dict(hit.payload or {})
            out.append(SearchResult(
                text=pl.get("text", ""),
                score=float(hit.score),
                metadata={k: v for k, v in pl.items() if k != "text"},
                method="dense",
            ))
        return out


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    # Implement RRF
    # 1. rrf_scores = {}  # text → {"score": float, "result": SearchResult}
    # 2. For each result_list in results_list:
    #      For rank, result in enumerate(result_list):
    #        rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    # 3. Sort by score descending
    # 4. Return top_k SearchResult with method="hybrid"
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    ordered = sorted(rrf_scores.values(), key=lambda e: e["score"], reverse=True)
    merged: list[SearchResult] = []
    for entry in ordered[:top_k]:
        r = entry["result"]
        md = dict(r.metadata) if isinstance(r.metadata, dict) else r.metadata
        merged.append(SearchResult(text=r.text, score=entry["score"], metadata=md, method="hybrid"))
    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
