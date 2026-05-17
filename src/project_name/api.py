"""still need to modify."""

from fastapi import FastAPI
import logging

logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
def root():
    """A simple root endpoint."""
    return {"message": "ok"}
