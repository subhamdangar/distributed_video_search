"""
Similarity Utilities
────────────────────
Cosine similarity computation using scikit-learn.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query embedding and a matrix of
    document embeddings.

    Args:
        query_embedding: 1-D array of shape (d,) or 2-D array of shape (1, d).
        doc_embeddings: 2-D array of shape (n, d).

    Returns:
        1-D array of shape (n,) with similarity scores in [-1, 1].
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    if doc_embeddings.ndim == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, doc_embeddings)
    return similarities.flatten()


def top_k_indices(scores: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Return indices of the top-k highest scores.

    Args:
        scores: 1-D array of similarity scores.
        k: Number of top results to return.

    Returns:
        Array of indices sorted by descending score.
    """
    k = min(k, len(scores))
    return np.argsort(scores)[::-1][:k]
