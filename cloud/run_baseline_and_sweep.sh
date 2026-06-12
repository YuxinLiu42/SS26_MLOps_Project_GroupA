#!/usr/bin/env bash
# Single-GPU "run everything" entrypoint for the Vertex container.
#
# Goal: acquire the scarce L4 ONCE, then run (optionally) the baseline
# fine-tune, an N-trial W&B sweep, and a by-subject evaluation of the sweep's
# best adapter — all in one GPU session. The job stays RUNNING throughout.
#
# Env vars:
#   SWEEP_COUNT        number of sweep trials (default 8)
#   WANDB_PROJECT      W&B project for the sweep (default scienceqa-paligemma2)
#   SKIP_BASELINE      "1" skips the baseline run (e.g. sweep #2: the 58.85%
#                      baseline already exists — don't retrain it)
#   SWEEP_ID           full entity/project/id of an EXISTING sweep to resume
#                      (preemption/restart recovery — keeps Bayesian history).
#                      Leave empty to register a fresh sweep from
#                      configs/sweep.yaml. Do NOT reuse an old sweep's id after
#                      changing the metric or the search space.
#   GCP_PROJECT        project for Secret Manager (default paligemma-scienceqa)
#   WANDB_SECRET_NAME  Secret Manager secret holding the W&B key (wandb-api-key)
#   HF_SECRET_NAME     Secret Manager secret holding the HF token (hf-token)
#   WANDB_API_KEY / HF_TOKEN  if already exported, Secret Manager is skipped
#
# Any extra args are forwarded to the BASELINE run as Hydra overrides, e.g.:
#   bash cloud/run_baseline_and_sweep.sh data.batch_size=4 data.num_workers=4
set -euo pipefail

SWEEP_COUNT="${SWEEP_COUNT:-8}"
WANDB_PROJECT="${WANDB_PROJECT:-scienceqa-paligemma2}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SWEEP_ID="${SWEEP_ID:-}"

# Secrets: fetched in-container from Secret Manager (names only in the job
# spec). Shared with entrypoint.sh — see cloud/fetch_secrets.sh.
source "$(dirname "$0")/fetch_secrets.sh"

echo ">>> fetching DVC-tracked data"
dvc pull -v

echo ">>> verifying CUDA is visible"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible - image likely has CPU-only torch'; print('CUDA OK:', torch.version.cuda)"

if [ "${SKIP_BASELINE}" = "1" ]; then
  echo ">>> [1/3] baseline skipped (SKIP_BASELINE=1)"
else
  echo ">>> [1/3] baseline fine-tune (run_name=baseline)"
  python -m project_name.train \
    trainer.wandb.enabled=true \
    trainer.wandb.run_name=baseline \
    "$@"
fi

echo ">>> [2/3] W&B sweep: ${SWEEP_COUNT} trials on project ${WANDB_PROJECT}"
if [ -n "${SWEEP_ID}" ]; then
  echo ">>> resuming existing sweep ${SWEEP_ID} (Bayesian history preserved)"
  SWEEP_PATH="${SWEEP_ID}"
else
  # Register the sweep and capture the 'wandb agent ENTITY/PROJECT/ID' line it
  # prints. tee keeps the full output in the job logs; '|| true' stops pipefail
  # from killing us if grep finds nothing (handled by the emptiness check).
  SWEEP_AGENT_CMD="$(wandb sweep --project "${WANDB_PROJECT}" configs/sweep.yaml 2>&1 \
    | tee /dev/stderr | grep -oE 'wandb agent [^[:space:]]+' | tail -1 || true)"
  if [ -z "${SWEEP_AGENT_CMD}" ]; then
    echo "!!! could not parse the sweep id from 'wandb sweep' output" >&2
    exit 1
  fi
  SWEEP_PATH="${SWEEP_AGENT_CMD#wandb agent }"
fi
echo ">>> launching: wandb agent ${SWEEP_PATH} --count ${SWEEP_COUNT}"
wandb agent "${SWEEP_PATH}" --count "${SWEEP_COUNT}"

# Evaluate the sweep's best run (by the sweep metric, val/accuracy max) while
# we still hold the GPU. Its adapter is still on local disk under
# checkpoints/adapter-<run_name>. ONE eval only: --by-subject already includes
# overall accuracy. Eval failures must not fail the job — the adapters are
# already uploaded to GCS by each run.
echo ">>> [3/3] evaluating the sweep's best adapter (--by-subject)"
(
  set -euo pipefail
  BEST_RUN="$(SWEEP_PATH="${SWEEP_PATH}" python - <<'PY'
import os

import wandb

api = wandb.Api()
sweep = api.sweep(os.environ["SWEEP_PATH"])
best = sweep.best_run()  # ranked by the sweep's own metric definition
print(best.name)
PY
)"
  echo ">>> best sweep run by val/accuracy: ${BEST_RUN}"
  ADAPTER_DIR="checkpoints/adapter-${BEST_RUN}"
  [ -d "${ADAPTER_DIR}" ] || { echo "!!! ${ADAPTER_DIR} not found on disk"; exit 1; }
  python -m project_name.evaluate "${ADAPTER_DIR}" --by-subject \
    --output-path eval_results.json
  if [ -n "${AIP_MODEL_DIR:-}" ]; then
    python - <<'PY'
import os
from pathlib import Path

from project_name.train import upload_to_gcs

uri = upload_to_gcs(Path("eval_results.json"), os.environ["AIP_MODEL_DIR"])
print(f"uploaded {uri}")
PY
  fi
) || echo "!!! best-adapter eval failed — adapters are already in GCS; run evaluate manually"
