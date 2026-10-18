"""LoRA fine-tuning pipeline — trains at multiple dataset sizes with proper evaluation."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

logger = structlog.get_logger()


@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir: str = "models"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    fp16: bool = True
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    seed: int = 42
    wandb_project: str = "ft-vs-prompt"


@dataclass
class TrainingResult:
    task: str
    train_size: int
    model_path: str
    train_loss: float
    eval_loss: float
    training_time_seconds: float
    gpu_hours: float
    peak_memory_gb: float
    steps: int


def prepare_model_and_tokenizer(config: TrainingConfig):
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def format_training_example(example: dict, tokenizer: Any, max_length: int) -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Follow the instruction exactly."},
        {"role": "user", "content": example["instruction"] + "\n\n" + example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train_lora(
    dataset: DatasetDict,
    train_size: int,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    if config is None:
        config = TrainingConfig()

    import time
    import wandb

    model, tokenizer = prepare_model_and_tokenizer(config)

    train_ds = dataset[f"train_{train_size}"]
    val_ds = dataset["validation"]

    def tokenize_fn(examples):
        return format_training_example(examples, tokenizer, config.max_seq_length)

    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names)

    task_name = train_ds[0].get("task", "unknown") if len(train_ds) > 0 else "unknown"
    run_name = f"{task_name}_size{train_size}_r{config.lora_r}"

    wandb.init(project=config.wandb_project, name=run_name, reinit=True)

    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}/{run_name}",
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=config.fp16,
        report_to="wandb",
        run_name=run_name,
        seed=config.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    t0 = time.perf_counter()
    trainer.train()
    training_time = time.perf_counter() - t0

    eval_results = trainer.evaluate()

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_memory = 0.0

    model_path = f"{config.output_dir}/{run_name}/final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    wandb.finish()

    result = TrainingResult(
        task=task_name,
        train_size=train_size,
        model_path=model_path,
        train_loss=float(trainer.state.log_history[-1].get("loss", 0)),
        eval_loss=float(eval_results.get("eval_loss", 0)),
        training_time_seconds=training_time,
        gpu_hours=training_time / 3600,
        peak_memory_gb=peak_memory,
        steps=trainer.state.global_step,
    )

    logger.info("training complete", task=result.task, size=train_size, gpu_hours=result.gpu_hours)
    return result


def load_trained_model(model_path: str, base_model: str = "meta-llama/Llama-3.2-3B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
