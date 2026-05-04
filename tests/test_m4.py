"""Tests for Module 4: Evaluation."""
import os
import sys
from numbers import Real
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, EvalResult


def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]


def test_evaluate_returns_metrics():
    """Unit test: không gọi RAGAS/OpenAI, không tải embedding (tránh treo vài phút im lặng)."""
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = pd.DataFrame(
        [
            {
                "question": "q",
                "answer": "a",
                "contexts": ["c"],
                "ground_truth": "gt",
                "faithfulness": 0.5,
                "answer_relevancy": 0.4,
                "context_precision": 0.8,
                "context_recall": 0.3,
            }
        ]
    )

    r = evaluate_ragas(
        ["q"],
        ["a"],
        [["c"]],
        ["gt"],
        _embeddings=MagicMock(),
        _evaluate_fn=lambda *args, **kwargs: fake_result,
    )
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r
        assert isinstance(r[k], Real) and not isinstance(r[k], bool)


@pytest.mark.slow
@pytest.mark.integration
def test_evaluate_ragas_live_smoke():
    """Tùy chọn: chạy RAGAS thật — cần API key + tải mô hình; chậm và ít log từ thư viện."""
    if not os.getenv("RUN_RAGAS_LIVE"):
        pytest.skip("Bật RUN_RAGAS_LIVE=1 nếu muốn kiểm tra end-to-end (chậm).")
    r = evaluate_ragas(["q"], ["a"], [["cửa"], ["sổ"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r


def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1


def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]
