"""Submit a Vertex AI custom training job for PaliGemma2-3B fine-tuning."""

from __future__ import annotations

import logging

from google.cloud import aiplatform
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("submit_train")

# GCP / project configuration
PROJECT_ID = "paligemma-scienceqa"
LOCATION = "europe-west4"
BUCKET = "mlops-paligemma-west4"
REPO = "mlops-images"
IMAGE_NAME = "paligemma-train"
IMAGE_TAG = "latest"

# Compute configuration: 1x NVIDIA L4 (24 GB) attached to a G2 machine.
MACHINE_TYPE = "g2-standard-8"
ACCELERATOR_TYPE = "NVIDIA_L4"
ACCELERATOR_COUNT = 1

# Secret NAMES only — the container resolves the values from Secret Manager
# at startup (cloud/fetch_secrets.sh), so the job spec never carries keys.
SECRET_ENV = [
    {"name": "GCP_PROJECT", "value": PROJECT_ID},
    {"name": "WANDB_SECRET_NAME", "value": "wandb-api-key"},
    {"name": "HF_SECRET_NAME", "value": "hf-token"},
]


def build_image_uri() -> str:
    """Assemble the full Artifact Registry image URI.

    Returns:
         The fully-qualified image URI that Vertex AI will pull.
    """
    registry = f"{LOCATION}-docker.pkg.dev"
    return f"{registry}/{PROJECT_ID}/{REPO}/{IMAGE_NAME}:{IMAGE_TAG}"


def main() -> None:
    """Configure and submit the custom training job."""
    image_uri = build_image_uri()

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=f"gs://{BUCKET}",
    )

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": MACHINE_TYPE,
                "accelerator_type": ACCELERATOR_TYPE,
                "accelerator_count": ACCELERATOR_COUNT,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": image_uri,
                "env": SECRET_ENV,
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name="paligemma-train",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=f"gs://{BUCKET}/vertex-runs",
    )

    logger.info("Image: %s", image_uri)
    logger.info(
        "Machine: %s + %dx %s", MACHINE_TYPE, ACCELERATOR_COUNT, ACCELERATOR_TYPE
    )
    logger.info("Output: gs://%s/vertex-runs", BUCKET)

    job.run(sync=True)


if __name__ == "__main__":
    main()
