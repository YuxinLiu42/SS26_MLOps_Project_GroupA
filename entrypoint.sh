set -euo pipefail
uv run --no-sync dvc pull -v
exec uv run --no-sync train "$@"
