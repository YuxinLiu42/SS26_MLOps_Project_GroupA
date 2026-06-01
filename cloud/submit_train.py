"""Submit a Vertex AI custom training job for PaliGemma2-3B fine-tuning."""

from __future__ import annotations

import logging
import os

from google.cloud import aiplatform
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("submit_train")

# GCP / project configuration
PROJECT_ID = "paligemma-scienceqa"
LOCATION = "europe-west3"
BUCKET = "mlops_paligemma"
REPO = "mlops-images"
IMAGE_NAME = "paligemma-train"
IMAGE_TAG = "latest"

# Compute configuration: 1x NVIDIA L4 (24 GB) attached to a G2 machine.
MACHINE_TYPE = "g2-standard-8"
ACCELERATOR_TYPE = "NVIDIA_L4"
ACCELERATOR_COUNT = 1

# Secrets expected in the local shell environment.
REQUIRED_SECRETS = ("WANDB_API_KEY", "HF_TOKEN")


def build_image_uri() -> str:
    """Assemble the full Artifact Registry image URI.

    Returns:
         The fully-qualified image URI that Vertex AI will pull.
    """
    registry = f"{LOCATION}-docker.pkg.dev"
    return f"{registry}/{PROJECT_ID}/{REPO}/{IMAGE_NAME}:{IMAGE_TAG}"


def collect_secret_env() -> list[dict[str, str]]:
    """Read required secrets from the environment for container injection.

    Returns:
        A list of {"name", "value"} dicts in the format Vertex expects.

    Raises:
        SystemExit: If any required secret is missing from the environment.
    """
    missing = [k for k in REQUIRED_SECRETS if not os.environ.get(k)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error("Export them first, e.g. `export WANDB_API_KEY=...`")
        raise SystemExit(1)
    return [{"name": k, "value": os.environ[k]} for k in REQUIRED_SECRETS]


def main() -> None:
    """Configure and submit the custom training job."""
    image_uri = build_image_uri()
    env = collect_secret_env()

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
                "env": env,
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
