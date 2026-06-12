set -euo pipefail
# Resolve W&B / HF credentials from Secret Manager when running on Vertex
# (no-op if WANDB_API_KEY / HF_TOKEN are already exported).
source cloud/fetch_secrets.sh
uv run --no-sync dvc pull -v
exec uv run --no-sync train "$@"
