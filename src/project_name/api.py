"""FastAPI inference service for PaliGemma2 ScienceQA predictions."""

import base64
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from project_name.model import PaliGemmaModule
from project_name.predict import load_model, predict_single

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)
log = logging.getLogger(__name__)

_module: PaliGemmaModule | None = None


def _fetch_gcs_dir(uri: str) -> Path:
    """Download a GCS directory (e.g. the production adapter) to a temp dir.

    Lets the serving container start from the stable GCS path
    (gs://mlops-paligemma-west4/models/production) instead of baking the
    adapter into the image — promotion then needs no rebuild/redeploy.

    Args:
        uri: gs://bucket/prefix directory holding the adapter files.

    Returns:
        Local directory containing the downloaded files.

    Raises:
        FileNotFoundError: If no objects exist under the prefix.
    """
    from google.cloud import storage

    parsed = urlparse(uri)
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    dest_root = Path(tempfile.mkdtemp(prefix="adapter-"))
    client = storage.Client()
    count = 0
    for blob in client.list_blobs(parsed.netloc, prefix=prefix):
        rel = blob.name[len(prefix) :]
        if not rel:
            continue
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        count += 1
    if not count:
        raise FileNotFoundError(f"No objects found under {uri}")
    log.info("Fetched %d adapter files from %s", count, uri)
    return dest_root


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the PaliGemma2 checkpoint on startup and release it on shutdown.

    The model path is resolved from the environment variable ``CHECKPOINT_PATH``
    and may be a LoRA adapter directory, a full .ckpt file, or a gs:// directory
    (downloaded to a temp dir first). If the variable is not set, or the path
    does not exist, the service starts without a loaded model and every
    /predict call will return 503 until the service is restarted with a valid
    path.

    Args:
        app: The FastAPI application instance (required by the lifespan protocol).
    """
    global _module
    checkpoint_env = os.environ.get("CHECKPOINT_PATH", "")
    if checkpoint_env:
        if checkpoint_env.startswith("gs://"):
            log.info("CHECKPOINT_PATH is a GCS uri — downloading %s", checkpoint_env)
            checkpoint_path = _fetch_gcs_dir(checkpoint_env)
        else:
            checkpoint_path = Path(checkpoint_env)
        if checkpoint_path.exists():
            log.info("Loading model from %s ...", checkpoint_path)
            _module = load_model(checkpoint_path)
            log.info("Model ready.")
        else:
            log.warning(
                "CHECKPOINT_PATH is set but file not found: %s", checkpoint_path
            )
    else:
        log.warning("CHECKPOINT_PATH not set — /predict will return 503.")

    yield

    _module = None
    log.info("Model released.")


app = FastAPI(
    title="PaliGemma2 ScienceQA API",
    description="Single-sample inference endpoint for "
    "PaliGemma2 fine-tuned on ScienceQA.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint.

    Attributes:
        question: The question text.
        choices: List of answer choice strings (2–10 items).
        hint: Optional hint text appended to the prompt.
        lecture: Optional lecture text appended to the prompt.
        image_b64: Required base64-encoded image string (JPEG or PNG).
        max_new_tokens: Maximum number of tokens the model may generate.
    """

    question: str = Field(..., min_length=1, description="Question text.")
    choices: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Answer choices, e.g. ['True', 'False'] or ['A', 'B', 'C'].",
    )
    hint: str = Field(default="", description="Optional hint text. Skipped if empty.")
    lecture: str = Field(
        default="", description="Optional lecture text. Skipped if empty."
    )
    image_b64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded image (JPEG or PNG).",
    )
    max_new_tokens: int = Field(
        default=10, ge=1, le=128, description="Max tokens to generate."
    )


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint.

    Attributes:
        prediction: Predicted answer letter, e.g. "A" or "B".
    """

    prediction: str = Field(..., description="Predicted answer letter (A/B/C/D/...).")


@app.get("/", summary="Health check")
def root() -> dict[str, str]:
    """Return service status and whether the model is loaded.

    Returns:
        A dict with ``status`` and ``model_loaded`` keys.
    """
    return {
        "status": "ok",
        "model_loaded": str(_module is not None),
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run inference on a single sample",
)
def predict(request: PredictRequest) -> PredictResponse:
    """Run single-sample inference and return the predicted answer letter.

    The image must be base64-encoded JPEG or PNG.
    The prediction is a single letter (A, B, C, ...) corresponding to the
    index of the predicted choice in ``request.choices``.

    Args:
        request: PredictRequest containing the question, choices, and
            optional image and hint.

    Returns:
        PredictResponse with the predicted answer letter.

    Raises:
        HTTPException 503: Model checkpoint has not been loaded.
        HTTPException 400: Image decoding failed.
    """
    if _module is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not loaded. "
            "Please set CHECKPOINT_PATH and restart the service.",
        )

    # Decode the image (required — PaliGemma is image-conditioned)
    try:
        image_bytes = base64.b64decode(request.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image_b64: {exc}",
        ) from exc

    # Only forward optional fields that were actually provided,
    # matching whatever columns survived --drop-cols during preprocessing.
    prompt_kwargs: dict[str, str] = {}
    if request.hint:
        prompt_kwargs["hint"] = request.hint
    if request.lecture:
        prompt_kwargs["lecture"] = request.lecture

    prediction = predict_single(
        module=_module,
        image=image,
        question=request.question,
        choices=request.choices,
        max_new_tokens=request.max_new_tokens,
        **prompt_kwargs,
    )

    log.info(
        "Prediction complete | question: %.60s | prediction: %s",
        request.question,
        prediction,
    )
    return PredictResponse(prediction=prediction)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
