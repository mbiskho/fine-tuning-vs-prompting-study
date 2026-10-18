"""Total Cost of Ownership model — full cost of fine-tuning counted properly."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CostComponents:
    gpu_hours: float
    gpu_cost_per_hour: float
    dataset_labour_hours: float
    labour_rate_per_hour: float
    serving_cost_per_1k_tokens: float
    base_model_tokens_per_request: int
    fine_tuned_tokens_per_request: int
    model_migration_frequency_months: int = 6


@dataclass
class TCOResult:
    method: str
    upfront_cost: float
    monthly_fixed_cost: float
    cost_per_1k_requests: float
    cost_at_1k_daily: float
    cost_at_10k_daily: float
    cost_at_100k_daily: float
    cost_at_1m_daily: float
    annual_cost_at_10k_daily: float
    notes: list[str] = field(default_factory=list)


@dataclass
class CrossoverAnalysis:
    task: str
    crossover_volume_daily: int
    crossover_volume_monthly: int
    prompting_below_crossover: str
    finetuning_above_crossover: str
    crossover_chart_data: dict[str, Any]
    recommendation: str


def compute_prompting_tco(
    cost_per_1k_tokens: float = 0.015,
    tokens_per_request: int = 500,
    requests_per_day: int = 10000,
) -> TCOResult:
    cost_per_request = (tokens_per_request / 1000) * cost_per_1k_tokens
    monthly_cost = cost_per_request * requests_per_day * 30

    return TCOResult(
        method="few_shot_prompting",
        upfront_cost=0.0,
        monthly_fixed_cost=0.0,
        cost_per_1k_requests=round(cost_per_request * 1000, 2),
        cost_at_1k_daily=round(cost_per_request * 1000 * 30, 2),
        cost_at_10k_daily=round(cost_per_request * 10000 * 30, 2),
        cost_at_100k_daily=round(cost_per_request * 100000 * 30, 2),
        cost_at_1m_daily=round(cost_per_request * 1000000 * 30, 2),
        annual_cost_at_10k_daily=round(cost_per_request * 10000 * 365, 2),
        notes=[
            f"Assumes {tokens_per_request} tokens/request at ${cost_per_1k_tokens}/1k tokens",
            "No upfront cost — pay per use",
            "Cost scales linearly with volume",
            "No model migration cost — provider handles updates",
        ],
    )


def compute_finetuning_tco(
    gpu_hours: float = 4.0,
    gpu_cost_per_hour: float = 1.50,
    dataset_labour_hours: float = 20.0,
    labour_rate_per_hour: float = 75.0,
    training_frequency_months: int = 6,
    serving_gpu_hours_per_month: float = 720,
    serving_gpu_cost_per_hour: float = 0.80,
    requests_per_day: int = 10000,
) -> TCOResult:
    dataset_cost = dataset_labour_hours * labour_rate_per_hour
    training_cost = gpu_hours * gpu_cost_per_hour
    upfront = dataset_cost + training_cost

    monthly_training = upfront / training_frequency_months
    monthly_serving = serving_gpu_hours_per_month * serving_gpu_cost_per_hour
    monthly_ops = 4 * labour_rate_per_hour

    monthly_fixed = monthly_serving + monthly_ops
    total_monthly = monthly_fixed + monthly_training

    cost_per_request = total_monthly / (requests_per_day * 30) if requests_per_day > 0 else 0

    return TCOResult(
        method="lora_fine_tuning",
        upfront_cost=round(upfront, 2),
        monthly_fixed_cost=round(monthly_fixed, 2),
        cost_per_1k_requests=round(cost_per_request * 1000, 2),
        cost_at_1k_daily=round(monthly_fixed + monthly_training, 2),
        cost_at_10k_daily=round(monthly_fixed + monthly_training, 2),
        cost_at_100k_daily=round(monthly_fixed + monthly_training, 2),
        cost_at_1m_daily=round(monthly_fixed + monthly_training, 2),
        annual_cost_at_10k_daily=round((monthly_fixed + monthly_training) * 12, 2),
        notes=[
            f"Dataset labour: {dataset_labour_hours}h × ${labour_rate_per_hour}/h = ${dataset_cost}",
            f"Training: {gpu_hours} GPU-h × ${gpu_cost_per_hour}/h = ${training_cost}",
            f"Retraining every {training_frequency_months} months adds ${monthly_training:.0f}/mo amortized",
            f"Serving infrastructure: ${monthly_serving:.0f}/mo (GPU 24/7)",
            f"Ops: ${monthly_ops:.0f}/mo",
            "Cost is mostly fixed — high volume improves unit economics",
        ],
    )


def find_crossover(
    prompting_tco: TCOResult,
    finetuning_tco: TCOResult,
    task: str,
    quality_delta: float = 0.0,
) -> CrossoverAnalysis:
    prompt_cost_per_request = prompting_tco.cost_per_1k_requests / 1000
    ft_fixed_monthly = finetuning_tco.monthly_fixed_cost + (finetuning_tco.upfront_cost / 6)

    if prompt_cost_per_request <= 0:
        return CrossoverAnalysis(
            task=task,
            crossover_volume_daily=999999,
            crossover_volume_monthly=99999999,
            prompting_below_crossover="Always cheaper",
            finetuning_above_crossover="Never cheaper",
            crossover_chart_data={},
            recommendation="Prompting is always cheaper — no crossover exists",
        )

    crossover_daily = int(ft_fixed_monthly / (30 * prompt_cost_per_request))

    recommendation = "Use prompting"
    if crossover_daily < 1000:
        recommendation = "Fine-tuning wins at any realistic volume"
    elif crossover_daily < 100000:
        recommendation = f"Fine-tuning wins above ~{crossover_daily:,} requests/day"
    else:
        recommendation = "Prompting wins at most realistic volumes"

    if quality_delta > 0.05:
        recommendation += " — quality advantage may justify fine-tuning at lower volumes"

    return CrossoverAnalysis(
        task=task,
        crossover_volume_daily=crossover_daily,
        crossover_volume_monthly=crossover_daily * 30,
        prompting_below_crossover="Cheaper below crossover",
        finetuning_above_crossover="Cheaper above crossover",
        crossover_chart_data={
            "volumes": [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
            "prompting_costs": [prompt_cost_per_request * v * 30 for v in [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]],
            "finetuning_costs": [ft_fixed_monthly for _ in range(9)],
        },
        recommendation=recommendation,
    )


def generate_decision_guide(analyses: list[CrossoverAnalysis]) -> str:
    lines = [
        "# Fine-Tuning vs Prompting: Decision Guide",
        "",
        "## When does fine-tuning beat a good prompt?",
        "",
        "The answer depends on three things: volume, quality requirements, and maintenance tolerance.",
        "",
        "## Crossover Volumes by Task",
        "",
        "| Task | Crossover (req/day) | Recommendation |",
        "|---|---|---|",
    ]

    for a in analyses:
        lines.append(f"| {a.task} | {a.crossover_volume_daily:,} | {a.recommendation} |")

    lines += [
        "",
        "## Decision Framework",
        "",
        "### Choose Prompting When:",
        "- Your request volume is below the crossover point",
        "- You need to iterate quickly (prompt changes take minutes, fine-tuning takes hours)",
        "- The base model is improving rapidly (you get the upgrades for free)",
        "- Your task definition is evolving",
        "- You don't have a labelled dataset",
        "",
        "### Choose Fine-Tuning When:",
        "- Your request volume is above the crossover point",
        "- You need the quality edge (even if it costs more)",
        "- Your task is stable and well-defined",
        "- You have a high-quality labelled dataset",
        "- Latency matters and you can serve a smaller model",
        "",
        "## Hidden Costs of Fine-Tuning",
        "",
        "1. **Dataset building**: 20-40 hours of labelling, cleaning, and validation",
        "2. **Model migration**: Every 6 months when the base model updates, you rebuild",
        "3. **Serving infrastructure**: You're now running a GPU instance 24/7",
        "4. **Monitoring drift**: Fine-tuned models degrade differently than prompted ones",
        "5. **Rollback complexity**: Can't just change a prompt back",
        "",
        "## The Retraining Burden",
        "",
        "If you fine-tune, you've bought a maintenance obligation:",
        "- Every base model release: re-label, re-train, re-evaluate",
        "- Every task change: re-label, re-train, re-evaluate",
        "- Every 6-12 months: check for drift, possibly re-train",
        "",
        "At $75/hour fully-loaded, the dataset labour alone is $1,500-$3,000 per rebuild.",
        "Over 2 years with quarterly base model updates, that's $12,000-$24,000 just in labelling.",
        "",
        "## Quality vs Cost Trade-off",
        "",
        "The benchmark win is usually real and usually irrelevant.",
        "A 2% F1 improvement on 10,000 requests/day might be worth it.",
        "A 1% improvement on 100 requests/day is noise.",
        "",
        "Below the crossover, fine-tuning loses on money even when it wins on the benchmark.",
    ]

    return "\n".join(lines)
