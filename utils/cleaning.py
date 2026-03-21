"""
Text Cleaning Utilities
───────────────────────
Minimal-but-effective cleaning for multilingual (Hindi/English/Hinglish) text.
Avoids aggressive cleaning that would destroy Hindi characters or code-mixed
sentences.
"""

import re
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Lightly clean input text while preserving Hindi/Devanagari characters.

    Steps:
      1. Strip leading/trailing whitespace
      2. Collapse multiple whitespace into single space
      3. Remove control characters (keeping newlines)
      4. Remove excessive newlines (keep max 2 consecutive)

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Strip edges
    text = text.strip()

    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Collapse multiple spaces (not newlines) into one
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def clean_query(query: str) -> str:
    """
    Clean a user query for processing.
    Lighter cleaning than document text — preserves question marks, etc.

    Args:
        query: Raw user query.

    Returns:
        Cleaned query string.
    """
    if not query:
        return ""

    query = query.strip()

    # Remove control characters
    query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)

    # Collapse whitespace
    query = re.sub(r'\s+', ' ', query)

    return query


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping chunks. Returns chunk content and
    character offsets for timestamp mapping.

    Args:
        text: Full text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of dicts with keys:
          - 'text': The chunk content
          - 'start_char': Starting character offset
          - 'end_char': Ending character offset
          - 'chunk_index': Zero-based chunk index
    """
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (period, question mark, etc.)
        if end < len(text):
            # Look for sentence-ending punctuation near the boundary
            boundary_search = text[end - 80:end + 20] if end > 80 else text[:end + 20]
            last_period = max(
                boundary_search.rfind('. '),
                boundary_search.rfind('? '),
                boundary_search.rfind('! '),
                boundary_search.rfind('। '),  # Hindi full stop (purna viram)
            )
            if last_period != -1:
                # Adjust end to the sentence boundary
                actual_end = (end - 80 + last_period + 2) if end > 80 else (last_period + 2)
                end = actual_end

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append({
                "text": chunk_content,
                "start_char": start,
                "end_char": min(end, len(text)),
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # Move start forward, accounting for overlap
        start = max(start + 1, end - overlap)

    return chunks
