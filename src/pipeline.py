"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report_with_extras
from src.m5_enrichment import enrich_chunks
from config import OPENAI_API_KEY, RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    timings: dict[str, float] = {}

    # Step 1: Load & Chunk (M1)
    print("\n[1/3] Chunking documents...")
    t0 = time.perf_counter()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents")
    timings["chunking_ms"] = (time.perf_counter() - t0) * 1000

    # Step 2: Enrichment (M5)
    print("\n[2/4] Enriching chunks (M5)...")
    t0 = time.perf_counter()
    enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    if enriched:
        # Use enriched text for indexing
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  Enriched {len(enriched)} chunks")
    else:
        print("  ⚠️  M5 not implemented — using raw chunks (fallback)")
    timings["enrichment_ms"] = (time.perf_counter() - t0) * 1000

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    t0 = time.perf_counter()
    search = HybridSearch()
    search.index(all_chunks)
    timings["indexing_ms"] = (time.perf_counter() - t0) * 1000

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker()
    timings["reranker_init_ms"] = (time.perf_counter() - t0) * 1000

    return search, reranker, timings


def generate_answer(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "Không tìm thấy thông tin."

    if not OPENAI_API_KEY:
        return contexts[0]

    try:
        from openai import OpenAI

        client = OpenAI()
        context_str = "\n\n".join(
            [f"[C{i+1}] {c}" for i, c in enumerate(contexts)]
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý trả lời câu hỏi. Quy tắc bắt buộc:\n"
                        "1) CHỈ dùng thông tin trong Context.\n"
                        "2) Nếu Context không đủ, trả lời đúng 1 câu: 'Không tìm thấy thông tin.'\n"
                        "3) Trả lời ngắn gọn, đúng trọng tâm.\n"
                        "4) Nếu có thể, trích dẫn nguồn bằng tag [C1], [C2]... ở cuối câu."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}",
                },
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return contexts[0]


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str], dict]:
    """Run single query through pipeline."""
    q_timings: dict[str, float] = {}

    t0 = time.perf_counter()
    results = search.search(query)
    q_timings["search_ms"] = (time.perf_counter() - t0) * 1000
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    q_timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    t0 = time.perf_counter()
    answer = generate_answer(query, contexts)
    q_timings["generate_ms"] = (time.perf_counter() - t0) * 1000
    return answer, contexts, q_timings


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    per_query_timings: list[dict] = []

    for i, item in enumerate(test_set):
        answer, contexts, q_timings = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        per_query_timings.append(q_timings)
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))

    latency_extras = {
        "latency_ms": {
            "search_avg_ms": float(mean([t.get("search_ms", 0.0) for t in per_query_timings])) if per_query_timings else 0.0,
            "rerank_avg_ms": float(mean([t.get("rerank_ms", 0.0) for t in per_query_timings])) if per_query_timings else 0.0,
            "generate_avg_ms": float(mean([t.get("generate_ms", 0.0) for t in per_query_timings])) if per_query_timings else 0.0,
        }
    }

    save_report_with_extras(results, failures, latency_extras)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, build_timings = build_pipeline()
    evaluate_pipeline(search, reranker)
    print("\n[Latency] Build breakdown (ms):")
    for k, v in build_timings.items():
        print(f"  {k}: {v:.1f}")
    print(f"\nTotal: {time.time() - start:.1f}s")
