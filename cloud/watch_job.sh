#!/usr/bin/env bash
# Watch a Vertex AI custom training job that uses Dynamic Workload Scheduler
# (Flex Start) to queue for a GPU, and react to its state:
#   QUEUED / PENDING -> Flex Start is waiting for capacity; keep polling
#   RUNNING          -> stream logs until the job ends
#   SUCCEEDED        -> list the checkpoint in GCS, then stop
#   FAILED / etc.    -> classify the error (capacity-timeout vs real bug), print, stop
#
# Flex Start replaces the old "create -> fail on stockout -> recreate" loop:
# one job queues itself until a GPU frees up (often off-peak / overnight),
# up to maxWaitDuration, then runs automatically. No MAX_RETRIES thrash.
set -uo pipefail

# On macOS, keep the machine awake for the whole watch, otherwise the laptop
# sleeps and this script is paused mid-night. Re-exec once under caffeinate.
if [ "$(uname -s)" = "Darwin" ] && [ -z "${UNDER_CAFFEINATE:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  echo ">>> macOS detected — re-running under 'caffeinate' to prevent sleep"
  export UNDER_CAFFEINATE=1
  exec caffeinate -is bash "$0" "$@"
fi

# settings to confirm / edit (env-overridable: the eval job reuses this
# watcher with TEMPLATE=cloud/vertex_eval.template.yaml etc.)
REGION=europe-west4
PROJECT=paligemma-scienceqa                   # project ID (custom-jobs create rejects the number)
DISPLAY_NAME="${DISPLAY_NAME:-paligemma-lora-train}"   # a timestamp is appended per run
TEMPLATE="${TEMPLATE:-cloud/vertex_config.template.yaml}"
RENDERED="${RENDERED:-cloud/vertex_config.yaml}"
STRATEGY=FLEX_START                           # Dynamic Workload Scheduler
MAX_WAIT="${MAX_WAIT:-auto}"                  # queue budget. 'auto' = 2x the queue time of
                                              # the last SUCCEEDED job, clamped to 6h-48h
                                              # (observed queues: 1.3h-19h). Or explicit,
                                              # e.g. MAX_WAIT=172800s. NOTE: 0s does NOT
                                              # mean indefinite — Vertex fails the job at
                                              # 24h (job 4498267958048456704 died that way)
POLL=30                                        # seconds between state checks
JOB=""                                         # populated by create_job

empty_count=0

# Stopping this script does NOT stop the Vertex job — remind how to cancel it.
cleanup() {
  echo ""
  echo ">>> watch interrupted."
  if [ -n "${JOB}" ]; then
    echo ">>> NOTE: the Vertex job keeps queuing/running on its own."
    echo ">>> cancel it with:"
    echo ">>>   gcloud ai custom-jobs cancel ${JOB} --region=${REGION} --project=${PROJECT}"
  fi
  exit 130
}
trap cleanup INT TERM

# fail fast before leaving this unattended
for bin in gcloud envsubst; do
  command -v "${bin}" >/dev/null 2>&1 || {
    echo "!!! '${bin}' not found in PATH. (macOS: 'brew install gettext' provides envsubst)"
    exit 1
  }
done
[ -f "${TEMPLATE}" ] || { echo "!!! template not found: ${TEMPLATE}"; exit 1; }

# Resolve MAX_WAIT=auto from history: 2x the queue time (createTime ->
# startTime) of the most recent SUCCEEDED job, clamped to 6h-48h. Falls back
# to 48h when no history is readable. Rationale: observed queues range from
# 1.3h to 19h, so a fixed 48h window can badly overshoot; sizing from the
# last success keeps the budget realistic without risking the 24h-cutoff trap.
derive_max_wait() {
  [ "${MAX_WAIT}" = "auto" ] || return 0
  local computed
  computed="$(gcloud ai custom-jobs list --region="${REGION}" --project="${PROJECT}" \
      --filter='state=JOB_STATE_SUCCEEDED' --sort-by=~createTime --limit=5 \
      --format='value(createTime,startTime)' 2>/dev/null \
    | python3 -c '
import sys
from datetime import datetime

def parse(t):
    return datetime.strptime(t.strip().rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S")

for line in sys.stdin:
    parts = line.split("\t")
    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        continue
    queue = (parse(parts[1]) - parse(parts[0])).total_seconds()
    print(int(max(6 * 3600, min(2 * queue, 48 * 3600))))
    break
')"
  if [ -n "${computed}" ]; then
    MAX_WAIT="${computed}s"
    echo ">>> MAX_WAIT=auto -> ${MAX_WAIT} (2x the last successful job's queue time, clamped 6h-48h)"
  else
    MAX_WAIT=172800s
    echo ">>> MAX_WAIT=auto but no job history readable -> default ${MAX_WAIT}"
  fi
}

# Make sure the rendered config carries a Flex Start schedule. If the template
# already defines one, leave it alone; otherwise append a top-level block.
ensure_scheduling_block() {
  if grep -qE '^[[:space:]]*scheduling:' "${RENDERED}"; then
    echo ">>> scheduling block already in template — leaving it as-is"
    return 0
  fi
  echo ">>> adding Flex Start schedule (strategy=${STRATEGY}, maxWaitDuration=${MAX_WAIT})"
  [ -n "$(tail -c1 "${RENDERED}")" ] && printf '\n' >> "${RENDERED}"   # ensure trailing newline
  cat >> "${RENDERED}" <<EOF
scheduling:
  strategy: ${STRATEGY}
  maxWaitDuration: ${MAX_WAIT}
EOF
}

create_job() {
  echo ">>> $(date +%T)  preflight + rendering config"
  derive_max_wait
  # Secrets stay in Secret Manager — the CONTAINER fetches them at startup
  # (run_baseline_and_sweep.sh), so the job spec carries only secret names.
  # Still verify they are readable now: an unreadable secret means the job
  # would 401 at runtime and waste a queued GPU slot.
  for s in wandb-api-key hf-token; do
    if ! gcloud secrets versions access latest --secret="${s}" --project="${PROJECT}" >/dev/null 2>&1; then
      echo "!!! cannot read secret '${s}' from Secret Manager."
      echo "!!! check the secret name and your access, then re-run."
      exit 1
    fi
  done

  # Pin the image by digest: Vertex resolves ':latest' at container START, so
  # after a long Flex Start queue the image could silently change.
  local image_base="europe-west4-docker.pkg.dev/${PROJECT}/mlops-images/paligemma-train"
  local digest
  digest="$(gcloud artifacts docker images describe "${image_base}:latest" \
    --format='value(image_summary.digest)' 2>/dev/null)"
  if [ -z "${digest}" ]; then
    echo "!!! could not resolve the digest of ${image_base}:latest — refusing to submit unpinned."
    exit 1
  fi
  export IMAGE_URI="${image_base}@${digest}"
  echo ">>> image pinned: ${IMAGE_URI}"

  # Sweep #2 knobs, forwarded into the container env (see the run script):
  #   SKIP_BASELINE=1  don't retrain the existing 58.85% baseline
  #   SWEEP_ID=e/p/id  resume an existing sweep after preemption (FRESH sweep
  #                    when empty — required after a metric/range change)
  export SKIP_BASELINE="${SKIP_BASELINE:-0}" SWEEP_ID="${SWEEP_ID:-}" \
    ADAPTER_GCS="${ADAPTER_GCS:-}"
  envsubst '${IMAGE_URI} ${SKIP_BASELINE} ${SWEEP_ID} ${ADAPTER_GCS}' \
    < "${TEMPLATE}" > "${RENDERED}"
  if [ -z "${SWEEP_ID}" ]; then
    # Vertex rejects env entries with an empty value ("Required field is not
    # set") — drop the SWEEP_ID entry (name + value lines) entirely; the run
    # script defaults it to empty anyway.
    sed -i '' -e '/- name: SWEEP_ID/{N;d;}' "${RENDERED}"
  fi
  if grep -q 'value: ""' "${RENDERED}"; then
    # Same Vertex rejection, generic case: a template variable rendered empty
    # (e.g. eval template submitted without ADAPTER_GCS). Fail before create.
    echo "!!! rendered config has env entries with an empty value (Vertex rejects those):"
    grep -B1 'value: ""' "${RENDERED}"
    exit 1
  fi
  ensure_scheduling_block

  # create, with a few retries for transient API hiccups (don't die on the first blip)
  local attempt=1 max_create=3
  while [ "${attempt}" -le "${max_create}" ]; do
    JOB="$(gcloud ai custom-jobs create \
      --region="${REGION}" --project="${PROJECT}" \
      --display-name="${DISPLAY_NAME}-$(date +%Y%m%d-%H%M%S)" \
      --config="${RENDERED}" \
      --format="value(name)" 2>/dev/null)"
    if [ -n "${JOB}" ]; then
      echo ">>> created job (Flex Start, will queue up to ${MAX_WAIT}):"
      echo ">>>   ${JOB}"
      return 0
    fi
    echo "!!! create attempt ${attempt}/${max_create} returned no job name — retrying in 15s"
    attempt=$((attempt + 1))
    sleep 15
  done
  echo "!!! could not create the job after ${max_create} attempts."
  echo "!!! run create manually once to see the real error:"
  echo "    gcloud ai custom-jobs create --region=${REGION} --project=${PROJECT} \\"
  echo "      --display-name=${DISPLAY_NAME} --config=${RENDERED}"
  exit 1
}

create_job

while true; do
  STATE="$(gcloud ai custom-jobs describe "${JOB}" \
    --region="${REGION}" --project="${PROJECT}" --format="value(state)" 2>/dev/null)"
  [ -n "${STATE}" ] && empty_count=0
  echo "$(date +%T)  ${STATE}"

  case "${STATE}" in
    JOB_STATE_RUNNING)
      echo ">>> RUNNING — streaming logs (blocks until the job ends)"
      # watch the first ~2 min for: dvc pull, model load (no 401), wandb run URL.
      # stream-logs may drop on a network blip; the loop just re-attaches.
      gcloud ai custom-jobs stream-logs "${JOB}" \
        --region="${REGION}" --project="${PROJECT}" || true
      ;;

    JOB_STATE_SUCCEEDED)
      echo ">>> SUCCEEDED — checking the checkpoint in GCS:"
      OUT="$(gcloud ai custom-jobs describe "${JOB}" \
        --region="${REGION}" --project="${PROJECT}" \
        --format="value(jobSpec.baseOutputDirectory.outputUriPrefix)" 2>/dev/null)"
      if [ -n "${OUT}" ]; then
        gcloud storage ls "${OUT%/}/**" --project="${PROJECT}" || true
      else
        echo "(no baseOutputDirectory on the job; check gs://mlops-paligemma-west4 manually)"
      fi
      break
      ;;

    JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_EXPIRED)
      ERR="$(gcloud ai custom-jobs describe "${JOB}" \
        --region="${REGION}" --project="${PROJECT}" --format="value(error.message)" 2>/dev/null)"
      echo ">>> ${STATE}: ${ERR:-<no error message>}"
      # best-effort: did Flex Start time out waiting for capacity, or is this a real bug?
      if echo "${ERR}" | grep -qiE "resources are insufficient|resources become available|wait duration|deadline exceeded|timeout"; then
        echo ">>> Looks like the Flex Start window (${MAX_WAIT}) expired before a ${REGION} GPU freed up."
        echo ">>> Options: raise MAX_WAIT (max 7d; 0s does NOT mean indefinite),"
        echo ">>> try an off-peak window, or just re-run this script to requeue."
      else
        echo ">>> This looks like a real error, not a capacity timeout. Full error object:"
        gcloud ai custom-jobs describe "${JOB}" \
          --region="${REGION}" --project="${PROJECT}" --format="value(error)" 2>/dev/null
      fi
      break
      ;;

    JOB_STATE_QUEUED|JOB_STATE_PENDING|JOB_STATE_CANCELLING|JOB_STATE_PAUSED|JOB_STATE_UPDATING)
      : # Flex Start is waiting for a GPU (QUEUED/PENDING) — keep polling
      ;;

    "")
      empty_count=$((empty_count + 1))
      echo ">>> empty state (transient API hiccup?) — will retry (${empty_count})"
      if [ "${empty_count}" -ge 10 ]; then
        echo ">>> describe returned empty ${empty_count} times in a row — confirm the job still exists:"
        echo ">>>   gcloud ai custom-jobs describe ${JOB} --region=${REGION} --project=${PROJECT}"
      fi
      ;;

    *)
      echo ">>> unexpected state '${STATE}' — will keep watching"
      ;;
  esac

  sleep "${POLL}"
done
