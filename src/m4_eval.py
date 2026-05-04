"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    df = result.to_pandas()
    
    per_question = []
    for _, row in df.iterrows():
        per_question.append(EvalResult(
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            contexts=row.get("contexts", []),
            ground_truth=row.get("ground_truth", ""),
            faithfulness=row.get("faithfulness", 0.0),
            answer_relevancy=row.get("answer_relevancy", 0.0),
            context_precision=row.get("context_precision", 0.0),
            context_recall=row.get("context_recall", 0.0)
        ))

    return {
        "faithfulness": result.get("faithfulness", 0.0),
        "answer_relevancy": result.get("answer_relevancy", 0.0),
        "context_precision": result.get("context_precision", 0.0),
        "context_recall": result.get("context_recall", 0.0),
        "per_question": per_question
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    failed = []
    for res in eval_results:
        # Handle nan values gracefully
        f = res.faithfulness if str(res.faithfulness) != 'nan' else 0.0
        ar = res.answer_relevancy if str(res.answer_relevancy) != 'nan' else 0.0
        cp = res.context_precision if str(res.context_precision) != 'nan' else 0.0
        cr = res.context_recall if str(res.context_recall) != 'nan' else 0.0
        
        avg_score = (f + ar + cp + cr) / 4.0
        
        metrics = {
            "faithfulness": f,
            "answer_relevancy": ar,
            "context_precision": cp,
            "context_recall": cr
        }
        
        worst_metric_name = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric_name]
        
        diagnosis = "Needs investigation"
        suggested_fix = "Manual review"
        
        if worst_metric_name == "faithfulness" and worst_score < 0.85:
            diagnosis = "LLM hallucinating"
            suggested_fix = "Tighten prompt, lower temperature"
        elif worst_metric_name == "context_recall" and worst_score < 0.75:
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking or add BM25"
        elif worst_metric_name == "context_precision" and worst_score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking or metadata filter"
        elif worst_metric_name == "answer_relevancy" and worst_score < 0.80:
            diagnosis = "Answer doesn't match question"
            suggested_fix = "Improve prompt template"
            
        failed.append({
            "question": res.question,
            "worst_metric": worst_metric_name,
            "score": worst_score,
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix
        })
        
    failed.sort(key=lambda x: x["avg_score"])
    
    return failed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


def save_report_with_extras(
    results: dict,
    failures: list[dict],
    extras: dict,
    path: str = "ragas_report.json",
):
    """Save evaluation report to JSON with optional extra fields."""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        **(extras or {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
