"""
Cache Agent (Semantic Cache)
────────────────────────────
Uses SQLite to cache previous query results with their embeddings.
On new queries, performs semantic similarity search against cached queries
to return cached results if a sufficiently similar query was seen before.

This drastically reduces redundant YouTube/web fetches.
"""

import json
import logging
import sqlite3
import numpy as np
from config.channels import DB_PATH, CACHE_SIMILARITY_THRESHOLD
from utils.similarity import compute_similarity

logger = logging.getLogger(__name__)


class CacheAgent:
    """
    Semantic cache backed by SQLite.

    Schema:
      - id: INTEGER PRIMARY KEY
      - query: TEXT (cleaned query string)
      - embedding: BLOB (numpy array serialized)
      - results: TEXT (JSON-serialized result object)
      - created_at: TIMESTAMP
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                embedding BLOB NOT NULL,
                results TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"CacheAgent: Database initialized at {self.db_path}")

    def search(self, query_embedding: np.ndarray, threshold: float = None) -> dict | None:
        """
        Search for a semantically similar cached query.

        Args:
            query_embedding: The embedding of the current query.
            threshold: Cosine similarity threshold for a cache hit.

        Returns:
            Cached result dict if found, else None.
        """
        threshold = threshold or CACHE_SIMILARITY_THRESHOLD
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, query, embedding, results FROM semantic_cache")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.info("CacheAgent: Cache is empty — no hit.")
            return None

        # Compare query embedding against all cached embeddings
        cached_embeddings = []
        for row in rows:
            cached_emb = np.frombuffer(row[2], dtype=np.float32)
            cached_embeddings.append(cached_emb)

        cached_embeddings = np.array(cached_embeddings)
        similarities = compute_similarity(query_embedding, cached_embeddings)

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        logger.info(f"CacheAgent: Best cache similarity = {best_score:.4f} "
                     f"(threshold = {threshold})")

        if best_score >= threshold:
            cached_query = rows[best_idx][1]
            cached_results = json.loads(rows[best_idx][3])
            logger.info(f"CacheAgent: HIT — cached query = '{cached_query}'")
            return {
                "cached_query": cached_query,
                "similarity": float(best_score),
                "results": cached_results,
            }

        logger.info("CacheAgent: MISS — no sufficiently similar cached query.")
        return None

    def store(self, query: str, embedding: np.ndarray, results: dict):
        """
        Store a query and its results in the cache.

        Args:
            query: The cleaned query string.
            embedding: The query embedding (numpy array).
            results: The result object to cache (must be JSON-serializable).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        embedding_blob = embedding.astype(np.float32).tobytes()
        results_json = json.dumps(results, ensure_ascii=False, default=str)

        cursor.execute(
            "INSERT INTO semantic_cache (query, embedding, results) VALUES (?, ?, ?)",
            (query, embedding_blob, results_json),
        )
        conn.commit()
        conn.close()
        logger.info(f"CacheAgent: Stored results for query '{query}'")

    def clear(self):
        """Clear all cached entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM semantic_cache")
        conn.commit()
        conn.close()
        logger.info("CacheAgent: Cache cleared.")
