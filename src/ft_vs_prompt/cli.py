"""CLI — ft-vs-prompt command-line interface."""

import json
from pathlib import Path

import click
import structlog

from .analysis.reporter import (
    StudyResult,
    export_full_report,
)
from .data.generator import (
    DataConfig,
    TaskType,
    generate_all_datasets,
)
from .eval.runner import (
    EvalConfig,
    EvalResult,
    run_prompting_eval,
    run_zero_shot_eval,
)
from .tco.model import (
    CrossoverAnalysis,
    TCOResult,
    compute_finetuning_tco,
    compute_prompting_tco,
    find_crossover,
    generate_decision_guide,
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@click.group()
@click.version_option()
def main():
    """Fine-Tuning vs Prompting Study — when does fine-tuning beat a good prompt?"""


@main.command()
@click.option("--output", default="data", help="Output directory")
@click.option("--train-sizes", default="100,500,2000", help="Comma-separated training sizes")
def generate(output, train_sizes):
    """Generate datasets for all three task types."""
    sizes = [int(s.strip()) for s in train_sizes.split(",")]
    configs = [
        DataConfig(task=TaskType.CLASSIFICATION, train_sizes=sizes, output_dir=output),
        DataConfig(task=TaskType.STRUCTURED_EXTRACTION, train_sizes=sizes, output_dir=output),
        DataConfig(task=TaskType.STYLE_CONSTRAINED_GENERATION, train_sizes=sizes, output_dir=output),
    ]
    results = generate_all_datasets(configs, output)

    for task, ds in results.items():
        click.echo(f"  {task}: {len(ds)} splits — {', '.join(ds.keys())}")


@main.command()
@click.option("--data-dir", default="data", help="Dataset directory")
@click.option("--tasks", default="classification,extraction,style_generation", help="Tasks to evaluate")
@click.option("--output", default="results", help="Results directory")
@click.option("--test-size", default=200, help="Number of test examples per task")
def evaluate_prompting(data_dir, tasks, output, test_size):
    """Evaluate prompting baselines on all tasks."""
    task_list = [t.strip() for t in tasks.split(",")]
    config = EvalConfig(test_size=test_size)

    results: list[StudyResult] = []

    for task in task_list:
        task_dir = Path(data_dir) / task
        test_file = task_dir / "test.jsonl"

        if not test_file.exists():
            click.echo(f"  Skipping {task}: no test data at {test_file}")
            continue

        from datasets import Dataset

        test_ds = Dataset.from_json(str(test_file))

        subtask_map = {
            "classification": "sentiment",
            "extraction": "contact",
            "style_generation": "formal",
        }

        click.echo(f"  Evaluating {task} (zero-shot)...")
        zs = run_zero_shot_eval(test_ds, task, subtask_map.get(task, ""), config=config)

        click.echo(f"  Evaluating {task} (few-shot)...")
        fs = run_prompting_eval(test_ds, task, subtask_map.get(task, ""), config=config)

        prompting_tco = compute_prompting_tco()
        finetuning_tco = compute_finetuning_tco()

        quality_delta = fs.accuracy - zs.accuracy if zs.accuracy > 0 else 0.0
        crossover = find_crossover(prompting_tco, finetuning_tco, task, quality_delta)

        study = StudyResult(
            task=task,
            prompting_result=fs,
            zero_shot_result=zs,
            prompting_tco=prompting_tco,
            finetuning_tco=finetuning_tco,
            crossover=crossover,
        )
        results.append(study)

        click.echo(f"    Few-shot:  acc={fs.accuracy:.3f}, p95={fs.p95_latency_ms:.0f}ms")
        click.echo(f"    Zero-shot: acc={zs.accuracy:.3f}, p95={zs.p95_latency_ms:.0f}ms")
        click.echo(f"    Crossover: {crossover.crossover_volume_daily:,} req/day")
        click.echo(f"    → {crossover.recommendation}")

    output_path = Path(output)
    export_full_report(results, output_path)
    click.echo(f"\nResults saved to {output_path}/")


@main.command()
@click.option("--data-dir", default="data", help="Dataset directory")
@click.option("--task", default="classification", help="Task type")
@click.option("--base-model", default="meta-llama/Llama-3.2-3B-Instruct", help="Base model")
@click.option("--train-sizes", default="100,500,2000", help="Training sizes")
@click.option("--output", default="models", help="Model output directory")
@click.option("--epochs", default=3, help="Training epochs")
def train(data_dir, task, base_model, train_sizes, output, epochs):
    """Run LoRA fine-tuning at multiple dataset sizes."""
    from .training.lora import TrainingConfig, train_lora
    from datasets import DatasetDict, Dataset

    sizes = [int(s.strip()) for s in train_sizes.split(",")]
    config = TrainingConfig(
        base_model=base_model,
        output_dir=output,
        num_epochs=epochs,
    )

    task_dir = Path(data_dir) / task
    datasets = DatasetDict()

    for size in sizes:
        train_file = task_dir / f"train_{size}.jsonl"
        if train_file.exists():
            datasets[f"train_{size}"] = Dataset.from_json(str(train_file))

    val_file = task_dir / "validation.jsonl"
    if val_file.exists():
        datasets["validation"] = Dataset.from_json(str(val_file))

    if len(datasets) == 0:
        click.echo(f"No datasets found in {task_dir}. Run 'ft-vs-prompt generate' first.", err=True)
        raise SystemExit(1)

    results = []
    for size in sizes:
        if f"train_{size}" not in datasets:
            click.echo(f"  Skipping size {size}: no training data")
            continue

        click.echo(f"  Training {task} with {size} examples...")
        result = train_lora(datasets, size, config)
        results.append(result)
        click.echo(f"    Loss: {result.eval_loss:.4f}, Time: {result.gpu_hours:.1f} GPU-h")

    result_path = Path(output) / f"{task}_training_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump([{
            "task": r.task, "train_size": r.train_size,
            "eval_loss": r.eval_loss, "gpu_hours": r.gpu_hours,
            "peak_memory_gb": r.peak_memory_gb, "model_path": r.model_path,
        } for r in results], f, indent=2)

    click.echo(f"\nTraining results saved to {result_path}")


@main.command()
@click.option("--output", default="docs", help="Output directory")
def decision_guide(output):
    """Generate the decision guide document."""
    tasks = ["classification", "structured_extraction", "style_constrained_generation"]

    analyses = []
    for task in tasks:
        prompting_tco = compute_prompting_tco()
        finetuning_tco = compute_finetuning_tco()
        crossover = find_crossover(prompting_tco, finetuning_tco, task)
        analyses.append(crossover)

    guide = generate_decision_guide(analyses)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "decision_guide.md", "w") as f:
        f.write(guide)

    click.echo(f"Decision guide written to {output_path}/decision_guide.md")


@main.command()
@click.option("--task", default="classification", help="Task type")
@click.option("--tokens-per-request", default=500, help="Tokens per request")
@click.option("--gpu-hours", default=4.0, help="Training GPU hours")
@click.option("--dataset-hours", default=20.0, help="Dataset creation hours")
def crossover(task, tokens_per_request, gpu_hours, dataset_hours):
    """Compute cost crossover for a specific scenario."""
    prompting_tco = compute_prompting_tco(tokens_per_request=tokens_per_request)
    finetuning_tco = compute_finetuning_tco(
        gpu_hours=gpu_hours,
        dataset_labour_hours=dataset_hours,
    )
    analysis = find_crossover(prompting_tco, finetuning_tco, task)

    click.echo(f"\n  Task: {task}")
    click.echo(f"  Prompting cost: ${prompting_tco.cost_per_1k_requests}/1K requests")
    click.echo(f"  Fine-tuning monthly fixed: ${finetuning_tco.monthly_fixed_cost:.0f}")
    click.echo(f"  Fine-tuning upfront: ${finetuning_tco.upfront_cost:.0f}")
    click.echo(f"")
    click.echo(f"  Crossover volume: {analysis.crossover_volume_daily:,} requests/day")
    click.echo(f"  Recommendation: {analysis.recommendation}")
    click.echo(f"")
    click.echo(f"  Prompting monthly costs:")
    click.echo(f"    1K/day:  ${prompting_tco.cost_at_1k_daily:.0f}")
    click.echo(f"    10K/day: ${prompting_tco.cost_at_10k_daily:.0f}")
    click.echo(f"    100K/day: ${prompting_tco.cost_at_100k_daily:.0f}")
    click.echo(f"    1M/day:  ${prompting_tco.cost_at_1m_daily:.0f}")
    click.echo(f"")
    click.echo(f"  Fine-tuning monthly costs:")
    click.echo(f"    1K/day:  ${finetuning_tco.cost_at_1k_daily:.0f}")
    click.echo(f"    10K/day: ${finetuning_tco.cost_at_10k_daily:.0f}")
    click.echo(f"    100K/day: ${finetuning_tco.cost_at_100k_daily:.0f}")
    click.echo(f"    1M/day:  ${finetuning_tco.cost_at_1m_daily:.0f}")


if __name__ == "__main__":
    main()
