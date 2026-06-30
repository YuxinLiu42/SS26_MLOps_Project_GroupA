FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.11-bookworm AS base

WORKDIR /workspace

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Install scipali's DEPENDENCIES only (--no-install-project); the project itself
# is installed from a prebuilt wheel below.
# --no-dev: training needs base + dvc only, not test/docs/lint tooling.
# --group data: provides the dvc[gs] CLI (build-time `dvc config` + `dvc pull`).
RUN uv sync --frozen --no-install-project --no-dev --group data

ENV VIRTUAL_ENV=/workspace/.venv

COPY src src/
COPY configs configs/
COPY cloud cloud/
COPY .dvc .dvc/
COPY data data/
COPY entrypoint.sh entrypoint.sh
COPY README.md README.md
COPY LICENSE LICENSE
COPY wheelhouse/ wheelhouse/

RUN mkdir -p models

# Install scipali from a PREBUILT wheel (built outside the image with `uv build`
# on the dev host and shipped in wheelhouse/). Building the project IN-image drops
# its subpackages -- only top-level `scipali` survives, so
# `python -m scipali.models.optimize` fails with
# `ModuleNotFoundError: No module named 'scipali.models'`. This was reproduced with
# both `uv sync` and `uv build`, even after pinning uv 0.11.6 via COPY --from, so
# it is a uv/uv_build behaviour we could not fix in-image. The locally-built wheel
# (verified to contain every subpackage) is the reliable path.
#   CAVEAT: a build from the bare git repo / CI has no wheelhouse and fails the
#   COPY above -- this image is built MANUALLY via /tmp staging that injects the
#   wheel (the mlops-ci-train trigger does not build it). Deps are installed
#   above, so install the wheel with --no-deps.
RUN uv pip install --no-deps --reinstall wheelhouse/*.whl
RUN uv pip install --no-cache-dir --reinstall torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu118

ENV PATH="/usr/local/nvidia/bin:/workspace/.venv/bin:$PATH"

# Vertex injects the GPU driver into /usr/local/nvidia/lib64 at runtime, but the
# uv/bookworm base never puts it on the library path — so torch can't find
# libcuda.so.1 and torch.cuda.is_available() is False even with a CUDA build.
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib"

# The image has no .git (.dockerignore excludes it), so DVC must run in "no SCM"
# mode or `dvc pull` errors looking for a git repo. Belt-and-suspenders with the
# copied .dvc/config so the image is correct regardless of the local setting.
RUN dvc config core.no_scm true
