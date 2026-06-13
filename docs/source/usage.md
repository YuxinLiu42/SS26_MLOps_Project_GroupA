# Usage

All commands assume the project environment is synced (`uv sync`).

## Data

```bash
# Download derek-thomas/ScienceQA (image questions) and preprocess the splits
uv run python -m project_name.data download
uv run python -m project_name.data preprocess --overwrite
dvc push   # publish processed data to the GCS remote
```

## Train (local)

```bash
uv run train trainer.wandb.enabled=true trainer.wandb.run_name=local-test
```

Hyperparameters live in `configs/` (Hydra). The learning rate is derived from
`model.base_learning_rate` via a sqrt batch-size rule unless set explicitly.

## Train + sweep on Vertex AI

```bash
# baseline + N-trial W&B sweep + by-subject eval, on one L4 (Flex Start queue)
bash cloud/watch_job.sh                         # full run
SKIP_BASELINE=1 bash cloud/watch_job.sh         # sweep only
```

## Evaluate an adapter

```bash
# local
uv run python -m project_name.evaluate checkpoints/adapter-production --by-subject
# standalone Vertex eval job against any GCS adapter
TEMPLATE=cloud/vertex_eval.template.yaml RENDERED=cloud/vertex_eval.yaml \
  DISPLAY_NAME=paligemma-eval \
  ADAPTER_GCS=gs://mlops-paligemma-west4/models/production \
  bash cloud/watch_job.sh
```

## Predict / serve

```bash
# single prediction
uv run python -m project_name.predict checkpoints/adapter-production \
  -q "What gas do plants absorb?" -c "oxygen,carbon dioxide,nitrogen" -i img.png

# API (local; PREDICT_DEVICE=cpu since MPS crashes on PaliGemma matmuls)
CHECKPOINT_PATH=checkpoints/adapter-production PREDICT_DEVICE=cpu \
  uv run uvicorn project_name.api:app --port 8000

# Streamlit frontend over the API
API_URL=http://localhost:8000 \
  uv run --group serving streamlit run src/project_name/frontend.py
```

## Deploy to Cloud Run

Build the API image (amd64) and deploy. The service reads its adapter from
`CHECKPOINT_PATH` (a `gs://` path), so promoting a new model needs no redeploy.

```bash
# 1. build + push the API image
gcloud builds submit --config=cloud/cloudbuild.api.yaml --project=paligemma-scienceqa .

# 2. deploy (CPU, scale-to-zero, lazy model load)
gcloud run deploy paligemma-api \
  --image europe-west4-docker.pkg.dev/paligemma-scienceqa/mlops-images/paligemma-api:latest \
  --region europe-west4 --project paligemma-scienceqa \
  --execution-environment gen2 \
  --memory 32Gi --cpu 8 \
  --timeout 3600 --concurrency 1 --max-instances 3 --min-instances 0 \
  --set-env-vars CHECKPOINT_PATH=gs://mlops-paligemma-west4/models/production,PREDICT_DEVICE=cpu,LAZY_LOAD=1 \
  --set-secrets HF_TOKEN=hf-token:latest \
  --service-account 581237630637-compute@developer.gserviceaccount.com \
  --allow-unauthenticated
```

Notes: `concurrency 1` keeps one heavy inference per instance (avoids OOM on the
3B model); `max-instances 3` lets overflow requests spin new instances instead
of returning 429. First call to each instance is slow (~160 s) — it downloads
the base model and loads on CPU; later calls are ~10–27 s.

## Ops

```bash
# data-drift report (Evidently)
uv run --group serving python -m project_name.monitoring
# load test the deployed API (locust)
uv run --group serving locust -f tests/load/locustfile.py \
  --headless -u 5 -r 1 -t 1m --host <cloud-run-url>
# BentoML serving
uv run --group serving bentoml serve project_name.bento_service:ScienceQAService
```
