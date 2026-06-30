r"""Streamlit frontend for the PaliGemma2 ScienceQA API.

Two ways to drive the ``/predict`` endpoint:

* **Ask your own** — upload an image, type a question + comma-separated choices
  (plus optional hint / lecture).
* **Pick from ScienceQA** — browse the project's own processed test split by
  *subject → topic*, load a real sample (image, question, choices, hint,
  lecture), predict, and compare against the ground-truth answer.

A *History* sidebar keeps recent questions so any of them can be re-run with a
single click (fast start).

Points at the API via the ``API_URL`` env var, so the same UI works against a
local server or the deployed Cloud Run service. The dataset picker reads the
on-disk processed split at ``DATASET_PATH`` (default
``data/processed/ScienceQA-IMG``).

This module imports no project code, so it is launched standalone via ``uvx`` in
its own environment (Streamlit's Starlette server pins a newer ``starlette``
than the FastAPI API does, so they can't share one venv):

    API_URL=http://localhost:8000 \\
      uvx --with requests --with pillow --with datasets \\
      streamlit run src/scipali/serving/frontend.py

(``--with datasets`` is only needed for the "Pick from ScienceQA" mode; "Ask
your own" works without it.)
"""

from __future__ import annotations

import base64
import io
import os
import random

import requests  # type: ignore[import-untyped]
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
DATASET_PATH = os.environ.get("DATASET_PATH", "data/processed/ScienceQA-IMG")
MAX_HISTORY = 10


def letter(idx: int) -> str:
    """Map a 0-based choice index to its answer letter (0 -> 'A')."""
    return chr(ord("A") + idx)


@st.cache_resource(show_spinner="Loading ScienceQA test split…")
def load_test_split():
    """Load the processed ScienceQA-IMG test split from disk, or return None.

    ``datasets`` is imported lazily and every failure is swallowed so the "Ask
    your own" mode still works when the package or the on-disk data is missing.
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        return None
    if not os.path.isdir(DATASET_PATH):
        return None
    try:
        return load_from_disk(DATASET_PATH)["test"]
    except Exception:  # noqa: BLE001 - any load failure just disables this mode
        return None


@st.cache_data(show_spinner="Indexing subjects / topics…")
def build_index(_ds) -> dict[str, dict[str, list[int]]]:
    """Map subject -> topic -> [row indices], reading only the small label columns.

    The leading underscore on ``_ds`` tells Streamlit not to hash the
    (unhashable) dataset object when memoizing.
    """
    index: dict[str, dict[str, list[int]]] = {}
    for i, (subj, top) in enumerate(zip(_ds["subject"], _ds["topic"])):
        topics = index.setdefault(subj or "(unknown)", {})
        topics.setdefault(top or "(none)", []).append(i)
    return index


@st.cache_data(show_spinner=False)
def question_previews(_ds) -> list[str]:
    """Question column as a plain list, for the sample-browser dropdown labels."""
    return list(_ds["question"])


def post_predict(payload: dict) -> str:
    """POST one sample to the API and return the predicted answer letter."""
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["prediction"]


def run_prediction(
    *,
    question: str,
    choices: list[str],
    hint: str,
    lecture: str,
    image_bytes: bytes,
    source: str,
    truth: str | None = None,
) -> None:
    """Call the API, render the image + result, compare to truth, push to history."""
    payload = {
        "question": question,
        "choices": choices,
        "hint": hint,
        "lecture": lecture,
        "image_b64": base64.b64encode(image_bytes).decode(),
    }
    st.image(Image.open(io.BytesIO(image_bytes)), width=240)
    with st.spinner("Querying the model (first call may be slow while it loads)…"):
        try:
            pred = post_predict(payload)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            return

    idx = ord(pred) - ord("A")
    answer = choices[idx] if 0 <= idx < len(choices) else "(out of range)"
    st.metric("Predicted answer", f"{pred} — {answer}")
    if truth is not None:
        if pred == truth:
            st.success(f"✅ Correct (ground truth {truth})")
        else:
            st.error(f"❌ Wrong — ground truth is {truth}")

    st.session_state.history.insert(
        0,
        {
            "question": question,
            "choices": choices,
            "hint": hint,
            "lecture": lecture,
            "image_b64": payload["image_b64"],
            "source": source,
            "pred": pred,
            "truth": truth,
        },
    )
    del st.session_state.history[MAX_HISTORY:]


# --- page setup -----------------------------------------------------------
st.set_page_config(page_title="ScienceQA · PaliGemma2", page_icon="🔬")
st.title("🔬 ScienceQA — PaliGemma2")
st.caption(f"Backend: {API_URL}")

st.session_state.setdefault("history", [])
st.session_state.setdefault("replay", None)

# Health badge so it's obvious whether the backend is up and the model loaded.
try:
    health = requests.get(f"{API_URL}/", timeout=5).json()
    if health.get("model_loaded") == "True":
        st.success("Backend up · model loaded")
    else:
        st.warning("Backend up · model not loaded yet (loads on first prediction)")
except requests.RequestException as exc:
    st.error(f"Backend unreachable: {exc}")

# --- sidebar: mode switch + history --------------------------------------
with st.sidebar:
    mode = st.radio("Mode", ["✍️ Ask your own", "📚 Pick from ScienceQA"])
    st.divider()
    st.subheader("History")
    if not st.session_state.history:
        st.caption("Recent questions appear here — click one to re-run it.")
    for i, h in enumerate(st.session_state.history):
        icon = "📚" if h["source"] == "dataset" else "✍️"
        if h["truth"] is None:
            mark = ""
        else:
            mark = " ✅" if h["pred"] == h["truth"] else " ❌"
        if st.button(
            f"{icon} {h['question'][:40]}… → {h['pred']}{mark}",
            key=f"hist-{i}",
            use_container_width=True,
        ):
            st.session_state.replay = h
            st.rerun()
    if st.session_state.history and st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# --- replay a previous question (fast start) -----------------------------
if st.session_state.replay is not None:
    h = st.session_state.replay
    st.session_state.replay = None
    st.subheader("Re-running a previous question")
    run_prediction(
        question=h["question"],
        choices=h["choices"],
        hint=h["hint"],
        lecture=h["lecture"],
        image_bytes=base64.b64decode(h["image_b64"]),
        source=h["source"],
        truth=h["truth"],
    )
    st.stop()

# --- mode: ask your own ---------------------------------------------------
if mode == "✍️ Ask your own":
    with st.form("ask"):
        image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
        question = st.text_input(
            "Question", "Which property do these objects have in common?"
        )
        choices_raw = st.text_input("Choices (comma-separated)", "soft, salty, sticky")
        hint = st.text_area("Hint (optional)", "")
        lecture = st.text_area("Lecture (optional)", "")
        submitted = st.form_submit_button("Predict")

    if submitted:
        if image_file is None:
            st.error("PaliGemma is image-conditioned — please upload an image.")
            st.stop()
        choices = [c.strip() for c in choices_raw.split(",") if c.strip()]
        if len(choices) < 2:
            st.error("Provide at least two choices.")
            st.stop()
        buf = io.BytesIO()
        Image.open(image_file).convert("RGB").save(buf, format="PNG")
        run_prediction(
            question=question,
            choices=choices,
            hint=hint,
            lecture=lecture,
            image_bytes=buf.getvalue(),
            source="custom",
        )

# --- mode: pick from ScienceQA -------------------------------------------
else:
    ds = load_test_split()
    if ds is None:
        st.info(
            f"Dataset picker unavailable — it needs the processed test split at "
            f"`{DATASET_PATH}` and the `datasets` package "
            "(launch with `uvx --with datasets …`). Use **Ask your own** instead."
        )
        st.stop()

    index = build_index(ds)
    previews = question_previews(ds)

    subject = st.selectbox("Subject", sorted(index))
    topic = st.selectbox("Topic", ["(all topics)"] + sorted(index[subject]))
    if topic == "(all topics)":
        candidates = sorted(i for ids in index[subject].values() for i in ids)
    else:
        candidates = index[subject][topic]
    st.caption(f"{len(candidates)} sample(s) in this subject / topic.")

    # Track the selected row; reset when the filter no longer contains it.
    if st.session_state.get("pick") not in candidates:
        st.session_state.pick = candidates[0]
    left, right = st.columns([1, 3])
    if left.button("🎲 Random", use_container_width=True):
        st.session_state.pick = random.choice(candidates)
    st.session_state.pick = right.selectbox(
        "Sample",
        candidates,
        index=candidates.index(st.session_state.pick),
        format_func=lambda i: f"#{i}: {previews[i][:60]}",
    )

    sample = ds[st.session_state.pick]
    choices = list(sample["choices"])
    truth = letter(sample["answer"])

    st.image(sample["image"], width=240)
    st.markdown(f"**Q:** {sample['question']}")
    st.markdown("\n".join(f"- **{letter(j)}.** {c}" for j, c in enumerate(choices)))
    if sample.get("hint"):
        st.caption(f"Hint: {sample['hint']}")
    with st.expander("Lecture / ground-truth answer"):
        if sample.get("lecture"):
            st.write(sample["lecture"])
        st.write(f"Ground-truth answer: **{truth}**")

    if st.button("Predict", type="primary"):
        buf = io.BytesIO()
        sample["image"].convert("RGB").save(buf, format="PNG")
        run_prediction(
            question=sample["question"],
            choices=choices,
            hint=sample.get("hint") or "",
            lecture=sample.get("lecture") or "",
            image_bytes=buf.getvalue(),
            source="dataset",
            truth=truth,
        )
