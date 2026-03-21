"""
Embedding Utilities
───────────────────
Loads the multilingual MiniLM model from sentence-transformers and provides
helper functions to embed single texts or batches.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 384-dimensional embeddings
  - Supports 50+ languages (Hindi, English, code-mixed)
  - Runs on CPU efficiently
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from config.channels import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Singleton model loader to avoid re-loading on every call
# ──────────────────────────────────────────────────────────────
_model = None


def get_model() -> SentenceTransformer:
    """
    Lazy-load and cache the embedding model.
    Returns the same model instance across all calls.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string.

    Args:
        text: Input text (Hindi / English / Hinglish).

    Returns:
        1-D numpy array of shape (384,).
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a batch of text strings.

    Args:
        texts: List of input strings.
        batch_size: Batch size for encoding (default 32).

    Returns:
        2-D numpy array of shape (len(texts), 384).
    """
    if not texts:
        return np.array([])

    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )
    return embeddings
