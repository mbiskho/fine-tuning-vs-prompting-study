import pytest

from ft_vs_prompt.prompting.engine import (
    PromptTemplate,
    build_classification_prompt,
    build_extraction_prompt,
    build_style_prompt,
    format_prompt,
    get_prompt_for_task,
    get_zero_shot_prompt,
)


class TestPromptEngine:
    def test_classification_prompt_sentiment(self):
        template = build_classification_prompt("sentiment")
        assert template.task == "classification"
        assert len(template.few_shot_examples) == 3
        assert "positive" in template.system_prompt.lower() or "sentiment" in template.system_prompt.lower()

    def test_classification_prompt_topic(self):
        template = build_classification_prompt("topic")
        assert template.task == "classification"
        assert "technology" in template.system_prompt.lower()

    def test_extraction_prompt(self):
        template = build_extraction_prompt("contact")
        assert template.task == "extraction"
        assert "json" in template.system_prompt.lower()

    def test_style_prompt(self):
        template = build_style_prompt("formal")
        assert template.task == "style_generation"
        assert "formal" in template.system_prompt.lower()

    def test_format_prompt(self):
        template = build_classification_prompt("sentiment")
        messages = format_prompt(template, "Great product!")

        assert len(messages) >= 1
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "Great product!" in messages[-1]["content"]

    def test_zero_shot_no_examples(self):
        template = get_zero_shot_prompt("classification", "sentiment")
        assert len(template.few_shot_examples) == 0

    def test_get_prompt_for_task(self):
        template = get_prompt_for_task("classification", "sentiment")
        assert template.task == "classification"
        assert len(template.few_shot_examples) == 3

    def test_unknown_task_falls_back(self):
        template = get_prompt_for_task("unknown", "")
        assert template.task == "classification"

    def test_all_subtasks_have_examples(self):
        for subtask in ["sentiment", "topic", "intent"]:
            template = build_classification_prompt(subtask)
            assert len(template.few_shot_examples) > 0

        for subtask in ["contact", "order"]:
            template = build_extraction_prompt(subtask)
            assert len(template.few_shot_examples) > 0

        for subtask in ["formal", "concise", "technical"]:
            template = build_style_prompt(subtask)
            assert len(template.few_shot_examples) > 0
