"""Tests for the FastAPI inference API."""

import base64
import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from project_name.api import app


@pytest.fixture(autouse=True)
def _no_checkpoint_env(monkeypatch):
    """Ensure CHECKPOINT_PATH is unset so lifespan doesn't load a real model."""
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)


def _make_image_b64() -> str:
    """Return a base64-encoded tiny PNG for requests that need a valid image."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


VALID_IMAGE_B64 = _make_image_b64()


def test_root_return_200() -> None:
    """Root endpoint returns HTTP 200 and expected keys."""
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_predict_missing_required_fields_returns_422() -> None:
    """Predict endpoint returns 422 when required fields are absent."""
    with TestClient(app) as client:
        response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_too_few_choices_returns_422() -> None:
    """Predict endpoint returns 422 when fewer than 2 choices are provided."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"question": "Is water wet?", "choices": ["Yes"]},
        )
    assert response.status_code == 422


def test_predict_without_model_returns_503() -> None:
    """Predict endpoint returns 503 when no checkpoint is loaded."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "question": "Is water wet?",
                "choices": ["Yes", "No"],
                "image_b64": VALID_IMAGE_B64,
            },
        )
    assert response.status_code == 503


def test_lifespan_fetches_gcs_checkpoint(monkeypatch, tmp_path) -> None:
    """A gs:// CHECKPOINT_PATH is downloaded via _fetch_gcs_dir, then loaded."""
    monkeypatch.setenv("CHECKPOINT_PATH", "gs://bucket/models/production")
    with (
        patch("project_name.api._fetch_gcs_dir", return_value=tmp_path) as mock_fetch,
        patch("project_name.api.load_model") as mock_load,
    ):
        with TestClient(app) as client:
            response = client.get("/")
        mock_fetch.assert_called_once_with("gs://bucket/models/production")
        mock_load.assert_called_once_with(tmp_path)
    assert response.json()["model_loaded"] == "True"


def test_lifespan_local_path_skips_gcs_fetch(monkeypatch, tmp_path) -> None:
    """A local CHECKPOINT_PATH never touches GCS."""
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path))
    with (
        patch("project_name.api._fetch_gcs_dir") as mock_fetch,
        patch("project_name.api.load_model") as mock_load,
    ):
        with TestClient(app):
            pass
        mock_fetch.assert_not_called()
        mock_load.assert_called_once()


def test_predict_returns_prediction() -> None:
    """Predict endpoint returns a prediction letter when model is loaded."""
    with (
        patch("project_name.api._module", new=object()),
        patch("project_name.api.predict_single", return_value="A"),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "question": "Is water wet?",
                    "choices": ["Yes", "No"],
                    "image_b64": VALID_IMAGE_B64,
                },
            )
    assert response.status_code == 200
    assert response.json()["prediction"] == "A"


def test_predict_with_hint_and_lecture() -> None:
    """Predict endpoint correctly forwards hint and lecture when provided."""
    with (
        patch("project_name.api._module", new=object()),
        patch("project_name.api.predict_single", return_value="B") as mock_predict,
    ):
        with TestClient(app) as client:
            client.post(
                "/predict",
                json={
                    "question": "What do plants absorb?",
                    "choices": ["Oxygen", "CO2"],
                    "hint": "Think about photosynthesis.",
                    "lecture": "Plants use sunlight to convert CO2.",
                    "image_b64": VALID_IMAGE_B64,
                },
            )
    # Verify hint and lecture were forwarded via prompt_kwargs
    _, kwargs = mock_predict.call_args
    assert kwargs.get("hint") == "Think about photosynthesis."
    assert kwargs.get("lecture") == "Plants use sunlight to convert CO2."


def test_predict_without_hint_and_lecture() -> None:
    """Predict endpoint does not forward hint or lecture when both are empty."""
    with (
        patch("project_name.api._module", new=object()),
        patch("project_name.api.predict_single", return_value="A") as mock_predict,
    ):
        with TestClient(app) as client:
            client.post(
                "/predict",
                json={
                    "question": "Is water wet?",
                    "choices": ["Yes", "No"],
                    "image_b64": VALID_IMAGE_B64,
                },
            )
    _, kwargs = mock_predict.call_args
    assert "hint" not in kwargs
    assert "lecture" not in kwargs


def test_predict_invalid_image_returns_400() -> None:
    """Predict endpoint returns 400 when image_b64 cannot be decoded."""
    with patch("project_name.api._module", new=object()):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "question": "Is water wet?",
                    "choices": ["Yes", "No"],
                    "image_b64": "not-valid-base64!!!",
                },
            )
    assert response.status_code == 400


def test_predict_missing_image_returns_422() -> None:
    """Predict endpoint returns 422 when image_b64 is absent."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"question": "Is water wet?", "choices": ["Yes", "No"]},
        )
    assert response.status_code == 422
