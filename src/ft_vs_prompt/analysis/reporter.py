"""Analysis — charts, crossover visualization, and summary reports."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import structlog

from ..eval.runner import EvalResult
from ..tco.model import CrossoverAnalysis, TCOResult, find_crossover, compute_finetuning_tco, compute_prompting_tco

logger = structlog.get_logger()


@dataclass
class StudyResult:
    task: str
    prompting_result: EvalResult | None = None
    zero_shot_result: EvalResult | None = None
    finetuning_results: list[EvalResult] = field(default_factory=list)
    prompting_tco: TCOResult | None = None
    finetuning_tco: TCOResult | None = None
    crossover: CrossoverAnalysis | None = None


def create_quality_comparison_chart(
    study_results: list[StudyResult],
    output_path: Path | None = None,
) -> go.Figure:
    fig = go.Figure()

    tasks = [r.task for r in study_results]
    x_positions = list(range(len(tasks)))

    for i, result in enumerate(study_results):
        if result.prompting_result:
            fig.add_trace(go.Bar(
                name="Few-Shot Prompting",
                x=[tasks[i]],
                y=[result.prompting_result.accuracy],
                marker_color="#2196F3",
                showlegend=(i == 0),
            ))

        if result.zero_shot_result:
            fig.add_trace(go.Bar(
                name="Zero-Shot",
                x=[tasks[i]],
                y=[result.zero_shot_result.accuracy],
                marker_color="#90CAF9",
                showlegend=(i == 0),
            ))

        for ft_result in result.finetuning_results:
            fig.add_trace(go.Bar(
                name=f"LoRA ({ft_result.train_size} ex)",
                x=[tasks[i]],
                y=[ft_result.accuracy],
                marker_color=["#4CAF50", "#FF9800", "#F44336"][min(ft_result.train_size, 2) // 500] if ft_result.train_size else "#4CAF50",
                showlegend=(i == 0),
            ))

    fig.update_layout(
        title="Quality Comparison: Prompting vs Fine-Tuning",
        barmode="group",
        xaxis_title="Task",
        yaxis_title="Accuracy / F1",
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def create_crossover_chart(
    study_results: list[StudyResult],
    output_path: Path | None = None,
) -> go.Figure:
    fig = go.Figure()

    for result in study_results:
        if result.crossover and result.crossover.crossover_chart_data:
            data = result.crossover.crossover_chart_data
            volumes = data["volumes"]

            fig.add_trace(go.Scatter(
                x=volumes,
                y=data["prompting_costs"],
                mode="lines+markers",
                name=f"{result.task} — Prompting",
                line=dict(dash="dash"),
            ))

            fig.add_trace(go.Scatter(
                x=volumes,
                y=data["finetuning_costs"],
                mode="lines",
                name=f"{result.task} — Fine-Tuning",
                fill="tonexty",
            ))

            fig.add_vline(
                x=result.crossover.crossover_volume_daily,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Crossover: {result.crossover.crossover_volume_daily:,}/day",
            )

    fig.update_layout(
        title="Cost Crossover: Prompting vs Fine-Tuning",
        xaxis_title="Requests per Day (log scale)",
        yaxis_title="Monthly Cost (USD)",
        xaxis_type="log",
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def create_tco_breakdown_chart(
    study_results: list[StudyResult],
    output_path: Path | None = None,
) -> go.Figure:
    fig = go.Figure()

    for result in study_results:
        if result.prompting_tco:
            fig.add_trace(go.Bar(
                name=f"{result.task} — Prompting",
                x=["1K", "10K", "100K", "1M"],
                y=[
                    result.prompting_tco.cost_at_1k_daily,
                    result.prompting_tco.cost_at_10k_daily,
                    result.prompting_tco.cost_at_100k_daily,
                    result.prompting_tco.cost_at_1m_daily,
                ],
                visible="legendonly" if len(study_results) > 1 else True,
            ))

        if result.finetuning_tco:
            fig.add_trace(go.Bar(
                name=f"{result.task} — Fine-Tuning",
                x=["1K", "10K", "100K", "1M"],
                y=[
                    result.finetuning_tco.cost_at_1k_daily,
                    result.finetuning_tco.cost_at_10k_daily,
                    result.finetuning_tco.cost_at_100k_daily,
                    result.finetuning_tco.cost_at_1m_daily,
                ],
            ))

    fig.update_layout(
        title="Monthly Cost by Volume",
        barmode="group",
        xaxis_title="Requests per Day",
        yaxis_title="Monthly Cost (USD)",
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def create_latency_comparison_chart(
    study_results: list[StudyResult],
    output_path: Path | None = None,
) -> go.Figure:
    fig = go.Figure()

    for result in study_results:
        if result.prompting_result:
            fig.add_trace(go.Bar(
                name=f"{result.task} — Prompting p95",
                x=[result.task],
                y=[result.prompting_result.p95_latency_ms],
                marker_color="#2196F3",
            ))

        for ft in result.finetuning_results:
            fig.add_trace(go.Bar(
                name=f"{result.task} — FT@{ft.train_size} p95",
                x=[result.task],
                y=[ft.p95_latency_ms],
                marker_color="#4CAF50",
            ))

    fig.update_layout(
        title="p95 Latency: Prompting vs Fine-Tuning",
        xaxis_title="Task",
        yaxis_title="p95 Latency (ms)",
        template="plotly_white",
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig


def export_summary_table(study_results: list[StudyResult], output_path: Path) -> pd.DataFrame:
    rows = []
    for r in study_results:
        if r.prompting_result:
            rows.append({
                "Task": r.task,
                "Method": "Few-Shot Prompting",
                "Train Size": "N/A",
                "Accuracy/F1": r.prompting_result.accuracy,
                "Exact Match": r.prompting_result.exact_match,
                "p95 Latency (ms)": r.prompting_result.p95_latency_ms,
                "Cost/1K": r.prompting_result.cost_per_1k,
            })

        if r.zero_shot_result:
            rows.append({
                "Task": r.task,
                "Method": "Zero-Shot",
                "Train Size": "N/A",
                "Accuracy/F1": r.zero_shot_result.accuracy,
                "Exact Match": r.zero_shot_result.exact_match,
                "p95 Latency (ms)": r.zero_shot_result.p95_latency_ms,
                "Cost/1K": r.zero_shot_result.cost_per_1k,
            })

        for ft in r.finetuning_results:
            rows.append({
                "Task": r.task,
                "Method": f"LoRA Fine-Tuned",
                "Train Size": ft.train_size,
                "Accuracy/F1": ft.accuracy,
                "Exact Match": ft.exact_match,
                "p95 Latency (ms)": ft.p95_latency_ms,
                "Cost/1K": ft.cost_per_1k,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path / "comparison_table.csv", index=False)
    df.to_markdown(output_path / "comparison_table.md", index=False)
    return df


def export_full_report(study_results: list[StudyResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    create_quality_comparison_chart(study_results, output_dir / "quality_comparison.html")
    create_crossover_chart(study_results, output_dir / "crossover_analysis.html")
    create_tco_breakdown_chart(study_results, output_dir / "tco_breakdown.html")
    create_latency_comparison_chart(study_results, output_dir / "latency_comparison.html")
    export_summary_table(study_results, output_dir)

    summary_data = []
    for r in study_results:
        entry: dict[str, Any] = {"task": r.task}

        if r.prompting_result:
            entry["few_shot"] = {
                "accuracy": r.prompting_result.accuracy,
                "f1": r.prompting_result.f1_score,
                "p95_ms": r.prompting_result.p95_latency_ms,
            }

        if r.zero_shot_result:
            entry["zero_shot"] = {
                "accuracy": r.zero_shot_result.accuracy,
                "f1": r.zero_shot_result.f1_score,
                "p95_ms": r.zero_shot_result.p95_latency_ms,
            }

        entry["fine_tuning"] = [{
            "train_size": ft.train_size,
            "accuracy": ft.accuracy,
            "f1": ft.f1_score,
            "p95_ms": ft.p95_latency_ms,
        } for ft in r.finetuning_results]

        if r.crossover:
            entry["crossover"] = {
                "volume_daily": r.crossover.crossover_volume_daily,
                "recommendation": r.crossover.recommendation,
            }

        if r.prompting_tco:
            entry["tco_prompting"] = asdict(r.prompting_tco)
        if r.finetuning_tco:
            entry["tco_finetuning"] = asdict(r.finetuning_tco)

        summary_data.append(entry)

    with open(output_dir / "full_report.json", "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    logger.info("report exported", path=str(output_dir))
