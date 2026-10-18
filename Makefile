.PHONY: help generate evaluate train crossover guide test clean lint format

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

install: ## Install in dev mode
	pip install -e ".[dev]"

generate: ## Generate all datasets
	ft-vs-prompt generate --output data

generate-small: ## Generate small test datasets
	ft-vs-prompt generate --output data --train-sizes 50,100,200

evaluate: ## Evaluate prompting baselines
	ft-vs-prompt evaluate-prompting --data-dir data --output results

train-classification: ## Train LoRA on classification
	ft-vs-prompt train --task classification --train-sizes 100,500,2000

train-extraction: ## Train LoRA on extraction
	ft-vs-prompt train --task extraction --train-sizes 100,500,2000

train-style: ## Train LoRA on style generation
	ft-vs-prompt train --task style_generation --train-sizes 100,500,2000

crossover: ## Compute cost crossover
	ft-vs-prompt crossover --task classification

guide: ## Generate decision guide
	ft-vs-prompt decision-guide --output docs

docker-train: ## Run training in Docker
	docker compose -f docker/docker-compose.yml --profile train run --rm train

docker-evaluate: ## Run evaluation in Docker
	docker compose -f docker/docker-compose.yml --profile evaluate run --rm evaluate

test: ## Run unit tests
	python -m pytest tests/unit -v

test-cov: ## Run tests with coverage
	python -m pytest tests/unit -v --cov=src/ft_vs_prompt --cov-report=term-missing

clean: ## Clean generated data and results
	rm -rf data/* results/* models/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint: ## Lint code
	python -m ruff check src/ tests/

format: ## Format code
	python -m ruff format src/ tests/

typecheck: ## Type check
	python -m mypy src/
