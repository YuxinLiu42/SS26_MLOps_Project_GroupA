### Overall goal of the project
The goal of the project is to develop techniques that improve reasoning accuracy using PaliGemma foundation model.

### What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)

### How to you intend to include the framework into your project
We plan on utilizing one of the strengths of the Transformers framework which is that it provides thousands of pretrained models to perform different tasks. As a starting point we intend to use some of the pretrained models on our data and then see how we can further improve from there.

### What data are you going to run on (initially, may change)
We are using the dataset:lmms-lab/ScienceQA

Each sample in the train and test set has the following information:



### What deep learning models do you expect to use
We use the VLM model PaliGemma.

````markdown
# project_name

a short description

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````

## Serving

The FastAPI service (`src/project_name/api.py`, image: `dockerfiles/api.dockerfile`)
serves single-sample ScienceQA predictions from the **production adapter**.

`CHECKPOINT_PATH` accepts a local adapter dir, a `.ckpt` file, or a `gs://` directory —
the stable production path is fetched at startup, so promoting a new adapter
(copy to GCS + W&B `production` alias) requires **no rebuild or redeploy**:

```bash
# local (model weights cached from HF; needs HF access for the gated base model)
CHECKPOINT_PATH=gs://mlops-paligemma-west4/models/production \
  uvicorn project_name.api:app --host 0.0.0.0 --port 8000
```

**Deployment decision (2026-06-12):** demo-grade serving runs locally or as a
container on demand, NOT as an always-on cloud endpoint. Rationale: PaliGemma2-3B
needs a GPU for interactive latency; an always-on L4 endpoint (Vertex endpoint or
Cloud Run w/ GPU) costs more than this course project justifies, and Cloud Run CPU
inference (~minutes/request) times out for real use. The `gs://` startup fetch
keeps the container cloud-ready: `gcloud run deploy --image <api image>
--set-env-vars CHECKPOINT_PATH=gs://mlops-paligemma-west4/models/production`
is the documented path if an always-on endpoint is ever needed.
