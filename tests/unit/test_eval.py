import pytest

from ft_vs_prompt.eval.runner import (
    EvalResult,
    evaluate_classification,
    evaluate_extraction,
    _extract_json,
)


class TestEvaluation:
    def test_classification_perfect(self):
        preds = ["positive", "negative", "neutral", "positive", "negative"]
        refs = ["positive", "negative", "neutral", "positive", "negative"]
        metrics = evaluate_classification(preds, refs)
        assert metrics["accuracy"] == 1.0
        assert metrics["exact_match"] == 1.0

    def test_classification_partial(self):
        preds = ["positive", "negative", "neutral", "positive", "negative"]
        refs = ["positive", "positive", "neutral", "negative", "neutral"]
        metrics = evaluate_classification(preds, refs)
        assert 0.0 < metrics["accuracy"] < 1.0

    def test_extraction_perfect(self):
        preds = [
            '{"name": "John", "email": "john@test.com", "phone": "555-1234"}',
            '{"name": "Jane", "email": "jane@test.com", "phone": null}',
        ]
        refs = [
            '{"name": "John", "email": "john@test.com", "phone": "555-1234"}',
            '{"name": "Jane", "email": "jane@test.com", "phone": null}',
        ]
        metrics = evaluate_extraction(preds, refs)
        assert metrics["exact_match"] == 1.0

    def test_extract_json_valid(self):
        text = 'Output: {"name": "Test", "email": "a@b.com"}'
        result = _extract_json(text)
        assert result == {"name": "Test", "email": "a@b.com"}

    def test_extract_json_invalid(self):
        text = "No JSON here"
        result = _extract_json(text)
        assert result == {}

    def test_eval_result_dataclass(self):
        result = EvalResult(
            task="classification", method="few_shot", train_size=None,
            num_examples=100, accuracy=0.95, f1_score=0.94, exact_match=0.90,
            mean_latency_ms=150.0, p95_latency_ms=300.0,
            total_cost=0.05, cost_per_1k=0.50,
        )
        assert result.accuracy == 0.95
        assert result.method == "few_shot"
