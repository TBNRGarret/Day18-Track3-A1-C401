"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # TODO: Load cross-encoder model
            # Option A: from FlagEmbedding import FlagReranker
            #           self._model = FlagReranker(self.model_name, use_fp16=True)
            # Option B: from sentence_transformers import CrossEncoder
            #           self._model = CrossEncoder(self.model_name)
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_name, use_fp16=True)
            except Exception:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        # TODO: Implement reranking
        # 1. model = self._load_model()
        # 2. pairs = [(query, doc["text"]) for doc in documents]
        # 3. scores = model.compute_score(pairs)  # FlagReranker
        #    OR scores = model.predict(pairs)      # CrossEncoder
        # 4. Combine: [(score, doc) for score, doc in zip(scores, documents)]
        # 5. Sort by score descending
        # 6. Return top_k RerankResult(text=..., original_score=doc["score"],
        #                              rerank_score=score, metadata=doc["metadata"], rank=i)
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, d.get("text", "")) for d in documents]

        scores = None
        if hasattr(model, "compute_score"):
            scores = model.compute_score(pairs)
        elif hasattr(model, "predict"):
            scores = model.predict(pairs)
        else:
            raise RuntimeError("Unsupported reranker model interface")

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: float(x[0]), reverse=True)
        scored_docs = scored_docs[:top_k]

        results: list[RerankResult] = []
        for i, (s, d) in enumerate(scored_docs, start=1):
            results.append(
                RerankResult(
                    text=d.get("text", ""),
                    original_score=float(d.get("score", 0.0)),
                    rerank_score=float(s),
                    metadata=d.get("metadata", {}),
                    rank=i,
                )
            )
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        try:
            from flashrank import RerankRequest

            model = self._load_model()
            passages = [{"id": i, "text": d.get("text", "")} for i, d in enumerate(documents)]
            rerank_results = model.rerank(RerankRequest(query=query, passages=passages))

            scored: list[tuple[float, int]] = []
            for r in rerank_results:
                # flashrank versions differ; support dict-like and attribute-like access
                if isinstance(r, dict):
                    rid = r.get("id")
                    score = r.get("score")
                else:
                    rid = getattr(r, "id", None)
                    score = getattr(r, "score", None)

                if rid is None:
                    continue
                scored.append((float(score or 0.0), int(rid)))

            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:top_k]

            out: list[RerankResult] = []
            for rank, (s, idx) in enumerate(scored, start=1):
                d = documents[idx]
                out.append(
                    RerankResult(
                        text=d.get("text", ""),
                        original_score=float(d.get("score", 0.0)),
                        rerank_score=float(s),
                        metadata=d.get("metadata", {}),
                        rank=rank,
                    )
                )
            return out
        except Exception:
            # Fallback: keep original order (stable), no external dependency required.
            out: list[RerankResult] = []
            for rank, d in enumerate(documents[:top_k], start=1):
                out.append(
                    RerankResult(
                        text=d.get("text", ""),
                        original_score=float(d.get("score", 0.0)),
                        rerank_score=float(d.get("score", 0.0)),
                        metadata=d.get("metadata", {}),
                        rank=rank,
                    )
                )
            return out


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    # TODO: Implement benchmark
    # 1. times = []
    # 2. for _ in range(n_runs):
    #      start = time.perf_counter()
    #      reranker.rerank(query, documents)
    #      times.append((time.perf_counter() - start) * 1000)  # ms
    # 3. return {"avg_ms": mean(times), "min_ms": min(times), "max_ms": max(times)}
    if n_runs <= 0:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)

    return {"avg_ms": float(mean(times)), "min_ms": float(min(times)), "max_ms": float(max(times))}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
