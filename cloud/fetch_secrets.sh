# Resolve WANDB_API_KEY / HF_TOKEN inside the Vertex container.
#
# Source this file (don't execute it): `. cloud/fetch_secrets.sh`
# Values are fetched from Secret Manager with the metadata-server token, so
# job specs carry only secret NAMES (previously the values sat in plaintext,
# visible via `gcloud ai custom-jobs describe`). Skipped for any variable that
# is already exported (e.g. local runs). Requires the job's service account to
# hold roles/secretmanager.secretAccessor on the secrets.
#
# Env vars (all optional):
#   GCP_PROJECT        project owning the secrets (default paligemma-scienceqa)
#   WANDB_SECRET_NAME  secret holding the W&B key (default wandb-api-key)
#   HF_SECRET_NAME     secret holding the HF token (default hf-token)

GCP_PROJECT="${GCP_PROJECT:-paligemma-scienceqa}"
WANDB_SECRET_NAME="${WANDB_SECRET_NAME:-wandb-api-key}"
HF_SECRET_NAME="${HF_SECRET_NAME:-hf-token}"

fetch_secret() {
  local name="$1" token
  token="$(curl -sf -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | python -c 'import sys, json; print(json.load(sys.stdin)["access_token"])')"
  curl -sf -H "Authorization: Bearer ${token}" \
    "https://secretmanager.googleapis.com/v1/projects/${GCP_PROJECT}/secrets/${name}/versions/latest:access" \
    | python -c 'import sys, json, base64; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())'
}

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo ">>> fetching ${WANDB_SECRET_NAME} from Secret Manager"
  WANDB_API_KEY="$(fetch_secret "${WANDB_SECRET_NAME}")"
  export WANDB_API_KEY
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo ">>> fetching ${HF_SECRET_NAME} from Secret Manager"
  HF_TOKEN="$(fetch_secret "${HF_SECRET_NAME}")"
  export HF_TOKEN
fi
