FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY config/ config/

RUN pip install --no-cache-dir -e .

ENV WANDB_MODE=offline

ENTRYPOINT ["ft-vs-prompt"]
CMD ["evaluate-prompting"]
