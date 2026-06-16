#!/usr/bin/env bash
# Standalone evaluation entrypoint for the Vertex container: evaluate ANY
# adapter from GCS on the ScienceQA test split, without training first.
#
# Usage (via watch_job.sh):
#   TEMPLATE=cloud/vertex_eval.template.yaml RENDERED=cloud/vertex_eval.yaml \
#   DISPLAY_NAME=paligemma-eval \
#   ADAPTER_GCS=gs://mlops-paligemma-west4/models/production \
#   bash cloud/watch_job.sh
#
# Env vars:
#   ADAPTER_GCS  (required) gs:// directory holding the LoRA adapter
#                (adapter_config.json + adapter_model.safetensors + ...)
#   GCP_PROJECT / *_SECRET_NAME  see cloud/fetch_secrets.sh (HF token is
#                needed to load the gated base model; W&B is not used here)
#
# Defaults to --batch-size 1 (serving-faithful, deterministic). Any extra args
# are forwarded to project_name.evaluate and override it (e.g. --batch-size 8).
set -euo pipefail

: "${ADAPTER_GCS:?set ADAPTER_GCS to the gs:// adapter directory}"

source "$(dirname "$0")/fetch_secrets.sh"

echo ">>> fetching DVC-tracked data"
dvc pull -v data/processed/ScienceQA-IMG.dvc   # only processed; raw is local-prep only

echo ">>> verifying CUDA is visible"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible - image likely has CPU-only torch'; print('CUDA OK:', torch.version.cuda)"

ADAPTER_DIR="checkpoints/eval-adapter"
echo ">>> downloading adapter from ${ADAPTER_GCS} to ${ADAPTER_DIR}"
ADAPTER_GCS="${ADAPTER_GCS}" ADAPTER_DIR="${ADAPTER_DIR}" python - <<'PY'
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from google.cloud import storage

uri = os.environ["ADAPTER_GCS"]
dest_root = Path(os.environ["ADAPTER_DIR"])
parsed = urlparse(uri)
# keep the path verbatim (minus leading slash): some adapters live under a
# double-slash prefix (vertex-output/paligemma-lora/model//adapter-<name>)
prefix = parsed.path.lstrip("/").rstrip("/") + "/"
client = storage.Client()
count = 0
for blob in client.list_blobs(parsed.netloc, prefix=prefix):
    rel = blob.name[len(prefix):]
    if not rel:
        continue
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))
    count += 1
print(f"downloaded {count} files")
sys.exit(0 if count else 1)
PY

echo ">>> evaluating ${ADAPTER_GCS} (--by-subject)"
# --batch-size 1: one sample at a time (no left-padding), matching how the API
# serves each /predict request — deterministic + serving-faithful. Placed before
# "$@" so a caller can still override it (click takes the last value).
python -m project_name.evaluate "${ADAPTER_DIR}" --by-subject \
  --batch-size 1 \
  --output-path eval_results.json "$@"

if [ -n "${AIP_MODEL_DIR:-}" ]; then
  python - <<'PY'
import os
from pathlib import Path

from project_name.train import upload_to_gcs

uri = upload_to_gcs(Path("eval_results.json"), os.environ["AIP_MODEL_DIR"])
print(f"uploaded {uri}")
PY
fi
