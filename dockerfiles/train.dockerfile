FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY .dvc .dvc/
COPY data data/
COPY entrypoint.sh entrypoint.sh
COPY README.md README.md
COPY LICENSE LICENSE

RUN mkdir -p models

RUN uv sync --frozen
RUN uv pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

ENTRYPOINT ["bash", "entrypoint.sh"]
