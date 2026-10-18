"""Evaluation — quality comparison between prompting and fine-tuning on held-out test sets."""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import structlog
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

logger = structlog.get_logger()


@dataclass
class EvalResult:
    task: str
    method: str
    train_size: int | None
    num_examples: int
    accuracy: float
    f1_score: float
    exact_match: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_cost: float
    cost_per_1k: float


@dataclass
class EvalConfig:
    test_size: int = 500
    max_new_tokens: int = 128
    temperature: float = 0.0
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"


def evaluate_classification(predictions: list[str], references: list[str]) -> dict[str, float]:
    pred_clean = [p.strip().lower().split("\n")[0].split(".")[0] for p in predictions]
    ref_clean = [r.strip().lower() for r in references]

    accuracy = accuracy_score(ref_clean, pred_clean)
    f1 = f1_score(ref_clean, pred_clean, average="weighted", zero_division=0)

    return {"accuracy": accuracy, "f1": f1, "exact_match": accuracy}


def evaluate_extraction(predictions: list[str], references: list[str]) -> dict[str, float]:
    exact_matches = 0
    partial_matches = 0

    for pred, ref in zip(predictions, references):
        try:
            pred_json = _extract_json(pred)
            ref_json = json.loads(ref)

            if pred_json == ref_json:
                exact_matches += 1

            if isinstance(pred_json, dict) and isinstance(ref_json, dict):
                shared = sum(1 for k in ref_json if k in pred_json and pred_json[k] == ref_json[k])
                if shared > 0:
                    partial_matches += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    n = len(predictions)
    return {
        "accuracy": partial_matches / n if n > 0 else 0.0,
        "f1": partial_matches / n if n > 0 else 0.0,
        "exact_match": exact_matches / n if n > 0 else 0.0,
    }


def evaluate_style(predictions: list[str], references: list[str]) -> dict[str, float]:
    return {
        "accuracy": 0.0,
        "f1": 0.0,
        "exact_match": 0.0,
    }


def _extract_json(text: str) -> Any:
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        return json.loads(match.group())
    return {}


def run_evaluation(
    test_dataset: Dataset,
    predict_fn: Callable[[str, str], str],
    task: str,
    method: str,
    train_size: int | None = None,
    cost_per_token: float = 0.0,
    config: EvalConfig | None = None,
) -> EvalResult:
    if config is None:
        config = EvalConfig()

    test_data = test_dataset.select(range(min(config.test_size, len(test_dataset))))

    predictions = []
    latencies = []
    total_tokens = 0

    for example in test_data:
        t0 = time.perf_counter()
        pred = predict_fn(example["instruction"], example["input"])
        latencies.append((time.perf_counter() - t0) * 1000)
        predictions.append(pred)
        total_tokens += len(pred.split())

    references = [example["output"] for example in test_data]

    if task == "classification":
        metrics = evaluate_classification(predictions, references)
    elif task == "extraction":
        metrics = evaluate_extraction(predictions, references)
    elif task == "style_generation":
        metrics = evaluate_style(predictions, references)
    else:
        metrics = {"accuracy": 0.0, "f1": 0.0, "exact_match": 0.0}

    latencies.sort()
    total_cost = total_tokens * cost_per_token

    return EvalResult(
        task=task,
        method=method,
        train_size=train_size,
        num_examples=len(test_data),
        accuracy=round(metrics["accuracy"], 4),
        f1_score=round(metrics["f1"], 4),
        exact_match=round(metrics["exact_match"], 4),
        mean_latency_ms=round(float(np.mean(latencies)), 1),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 1),
        total_cost=round(total_cost, 4),
        cost_per_1k=round(total_cost / len(test_data) * 1000, 4),
    )


def run_prompting_eval(
    test_dataset: Dataset,
    task: str,
    subtask: str,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    config: EvalConfig | None = None,
) -> EvalResult:
    from ..prompting.engine import format_prompt, get_prompt_for_task, get_zero_shot_prompt

    template = get_prompt_for_task(task, subtask)

    def predict(instruction: str, input_text: str) -> str:
        messages = format_prompt(template, input_text)
        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=config.temperature if config else 0.0,
                max_tokens=config.max_new_tokens if config else 128,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    return run_evaluation(test_dataset, predict, task, "few_shot_prompting", cost_per_token=0.000015, config=config)


def run_zero_shot_eval(
    test_dataset: Dataset,
    task: str,
    subtask: str,
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    config: EvalConfig | None = None,
) -> EvalResult:
    from ..prompting.engine import format_prompt, get_zero_shot_prompt

    template = get_zero_shot_prompt(task, subtask)

    def predict(instruction: str, input_text: str) -> str:
        messages = format_prompt(template, input_text)
        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=config.temperature if config else 0.0,
                max_tokens=config.max_new_tokens if config else 128,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    return run_evaluation(test_dataset, predict, task, "zero_shot_prompting", cost_per_token=0.000015, config=config)


def run_finetuned_eval(
    test_dataset: Dataset,
    task: str,
    model_path: str,
    train_size: int,
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    config: EvalConfig | None = None,
) -> EvalResult:
    from ..training.lora import load_trained_model

    model, tokenizer = load_trained_model(model_path, base_model)
    model.eval()

    def predict(instruction: str, input_text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow the instruction exactly."},
            {"role": "user", "content": instruction + "\n\n" + input_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens if config else 128,
                temperature=config.temperature if config else 0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    import torch
    return run_evaluation(test_dataset, predict, task, f"lora_ft_{train_size}", train_size, cost_per_token=0.0, config=config)
