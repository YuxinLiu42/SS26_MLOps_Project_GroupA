# Usage

All commands assume the project environment is synced (`uv sync`).

## Data

```bash
# Download derek-thomas/ScienceQA (image questions) and preprocess the splits
uv run python -m scipali.data.data download
uv run python -m scipali.data.data preprocess --overwrite
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
uv run python -m scipali.models.evaluate checkpoints/adapter-production --by-subject
# standalone Vertex eval job against any GCS adapter
TEMPLATE=cloud/vertex_eval.template.yaml RENDERED=cloud/vertex_eval.yaml \
  DISPLAY_NAME=paligemma-eval \
  ADAPTER_GCS=gs://mlops-paligemma-west4/models/production \
  bash cloud/watch_job.sh
```

## Optimize: quantization + pruning (M31)

Inference-optimization experiments on an L4 (results feed `reports/RESULTS.md`).
Run on Vertex via `cloud/run_optimize.sh` + `watch_job.sh`:

```bash
# quantization benchmark: bf16 vs int4 (bitsandbytes) vs bf16+torch.compile
TEMPLATE=cloud/vertex_optimize.template.yaml RENDERED=cloud/vertex_optimize.yaml \
  DISPLAY_NAME=paligemma-optimize \
  ADAPTER_GCS=gs://mlops-paligemma-west4/models/production \
  bash cloud/watch_job.sh

# weight-pruning sweep: accuracy vs sparsity over the full test split
# (SKIP_BENCHMARK=1 runs prune-only; fits a 32GB host via a histogram threshold)
SKIP_BENCHMARK=1 PRUNE_SPARSITIES=0.0,0.3,0.5,0.7 PRUNE_N_BATCHES=0 \
  TEMPLATE=cloud/vertex_optimize.template.yaml RENDERED=cloud/vertex_optimize.yaml \
  DISPLAY_NAME=paligemma-prune \
  ADAPTER_GCS=gs://mlops-paligemma-west4/models/production \
  bash cloud/watch_job.sh
```

The underlying Typer CLI (CUDA-only; runs inside the L4 container):

```bash
python -m scipali.models.optimize benchmark <adapter_dir> --output-path optimize_results.json
python -m scipali.models.optimize prune-sweep <adapter_dir> \
  --sparsities 0.0,0.3,0.5,0.7 --output-path prune_results.json
```

> A `prune-finetune` command (prune → masked fine-tune to recover accuracy) also
> exists, but it is **explored, not a deliverable**: pruning is an inference-time
> technique, and a short fine-tune recovered only ~1 pt — so we report the
> one-shot accuracy-vs-sparsity curve instead.

> **Build note:** the train/optimize image installs a locally-built wheel, so it
> is built **manually** via `/tmp` staging (a bare `gcloud builds submit` from the
> iCloud working dir stalls). The `mlops-ci-train` trigger is therefore disabled;
> `mlops-ci-api` still builds the API image on push.

## Predict / serve

```bash
# single prediction
uv run python -m scipali.serving.predict checkpoints/adapter-production \
  -q "What gas do plants absorb?" -c "oxygen,carbon dioxide,nitrogen" -i img.png

# API (local; PREDICT_DEVICE=cpu since MPS crashes on PaliGemma matmuls)
CHECKPOINT_PATH=checkpoints/adapter-production PREDICT_DEVICE=cpu \
  uv run uvicorn scipali.serving.api:app --port 8000

# Streamlit frontend over the API — launched standalone via uvx (it imports no
# project code, and Streamlit's Starlette server conflicts with FastAPI's
# pinned starlette, so it is kept out of the project env on purpose).
# Two modes: "Ask your own" (type a question) and "Pick from ScienceQA" (browse
# the processed test split by subject/topic and compare to ground truth). The
# dataset picker needs `--with datasets` + the local processed split; "Ask your
# own" works without either.
API_URL=http://localhost:8000 \
  uvx --with requests --with pillow --with datasets \
  streamlit run src/scipali/serving/frontend.py
```

Point `API_URL` at the deployed Cloud Run URL instead to drive the live service.

There is also a shell demo of the deployed API (health → predict → drift):

```bash
./cloud/demo_api.sh                              # auto-uses test sample 0
./cloud/demo_api.sh img.png "Question?" "a,b,c"  # bring your own sample
```

## Container images

Three Dockerfiles live in `dockerfiles/`: `train.dockerfile`, `api.dockerfile`,
and `predict.dockerfile`. Build them locally with the Invoke task:

```bash
inv docker-build        # builds train + api images locally (see tasks.py)
```

The images are **continuously built and verified in CI**: the Cloud Build
triggers `mlops-ci-api` and `mlops-ci-train` rebuild `api.dockerfile` and
`train.dockerfile` (amd64) on every push to `main` that touches
`src/scipali/**` or the build configs. A failing Dockerfile breaks the
build, so "the Dockerfiles build and work as intended" is enforced on
every change — no manual local build is required, though `inv docker-build`
remains available for local iteration.

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
uv run --group serving python -m scipali.monitoring.monitoring
# load test the deployed API (locust)
uv run --group serving locust -f tests/load/locustfile.py \
  --headless -u 5 -r 1 -t 1m --host <cloud-run-url>
# BentoML serving
uv run --group serving bentoml serve scipali.serving.bento_service:ScienceQAService
```
