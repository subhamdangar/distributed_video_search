"""
Query Understanding Agent
─────────────────────────
Responsible for:
  1. Cleaning the raw user query
  2. Generating a multilingual embedding for the query

This is the first agent in the pipeline — every query passes through here.
"""

import logging
import numpy as np
from utils.cleaning import clean_query
from utils.embeddings import embed_text
# 
logger = logging.getLogger(__name__)


class QueryAgent:
    """
    Processes raw user input into a cleaned query + dense embedding vector.
    Supports Hindi, English, and Hinglish (code-mixed) queries.
    """

    def process(self, raw_query: str) -> dict:
        """
        Process a raw query string.

        Args:
            raw_query: The user's natural-language query.

        Returns:
            Dictionary with keys:
              - 'original': The raw query as received
              - 'cleaned': The cleaned query string
              - 'embedding': numpy array of shape (384,)
        """
        logger.info(f"QueryAgent: Processing query — '{raw_query}'")

        cleaned = clean_query(raw_query)
        if not cleaned:
            logger.warning("QueryAgent: Query is empty after cleaning.")
            return {
                "original": raw_query,
                "cleaned": "",
                "embedding": np.zeros(384),
            }

        logger.info(f"QueryAgent: Cleaned query — '{cleaned}'")

        # Generate multilingual embedding
        embedding = embed_text(cleaned)
        logger.info(f"QueryAgent: Embedding generated — shape {embedding.shape}")

        return {
            "original": raw_query,
            "cleaned": cleaned,
            "embedding": embedding,
        }
