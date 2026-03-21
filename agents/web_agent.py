"""
Web Search Agent (Fallback)
───────────────────────────
Activated when no relevant YouTube results are found.

Pipeline:
  1. Search DuckDuckGo for the query
  2. Take top URLs
  3. Load page content using LangChain WebBaseLoader
  4. Clean, chunk, and embed
  5. Rank by similarity (with trusted-site boosting)
  6. Return the best result

Uses NO paid APIs — relies entirely on DuckDuckGo.
Handles Hinglish/Hindi queries by retrying with extracted English keywords.
"""

import logging
import re
import traceback
import time

import numpy as np

from config.channels import (
    WEB_SEARCH_MAX_URLS,
    TRUSTED_WEBSITES,
    TRUSTED_SITE_BOOST,
    TRANSCRIPT_CHUNK_SIZE,
    TRANSCRIPT_CHUNK_OVERLAP,
)
from utils.cleaning import clean_text, chunk_text
from utils.embeddings import embed_texts
from utils.similarity import compute_similarity, top_k_indices

logger = logging.getLogger(__name__)


def _extract_english_keywords(query: str) -> str:
    """
    Extract English/ASCII words from a Hinglish query for search retry.
    e.g., "binary search tree kaise banate hain" → "binary search tree"
    Also adds "tutorial" if the query looks educational.
    """
    # Keep only ASCII alphabetical words (strips Hindi/Devanagari)
    english_words = re.findall(r'[a-zA-Z]{2,}', query)

    # Filter out common Hindi words written in English
    hindi_stopwords = {
        "kaise", "kya", "hai", "hain", "kare", "karo", "ko", "ka", "ke",
        "ki", "mein", "se", "ye", "wo", "aur", "ya", "par", "pe",
        "banaye", "banate", "samjhao", "samjhaiye", "batao", "bataiye",
        "sikhe", "sikhiye", "kaise", "kyun", "kab", "kahan",
    }

    filtered = [w for w in english_words if w.lower() not in hindi_stopwords]

    if not filtered:
        return query  # Return original if nothing extracted

    result = " ".join(filtered)

    # Add "tutorial" for educational context if not present
    if "tutorial" not in result.lower() and "explained" not in result.lower():
        result += " tutorial"

    return result


def search_duckduckgo(query: str, max_results: int = None) -> list[dict]:
    """
    Search DuckDuckGo and return top results.
    If the original query returns 0 results (common with Hinglish),
    retries with extracted English keywords.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with 'title', 'url', 'snippet'.
    """
    max_results = max_results or WEB_SEARCH_MAX_URLS

    def _do_search(q: str) -> list[dict]:
        """Execute a single DuckDuckGo search using the ddgs package."""
        try:
            from ddgs import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", ""),
                    })
            return results
        except ImportError:
            # Fallback to old package name
            try:
                from duckduckgo_search import DDGS

                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(q, max_results=max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("href", r.get("link", "")),
                            "snippet": r.get("body", ""),
                        })
                return results
            except Exception as e:
                logger.debug(f"WebAgent: DuckDuckGo search error (fallback) for '{q}': {e}")
                return []
        except Exception as e:
            logger.debug(f"WebAgent: DuckDuckGo search error for '{q}': {e}")
            return []

    # ── Attempt 1: Original query ─────────────────────────────
    results = _do_search(query)
    if results:
        logger.info(f"WebAgent: DuckDuckGo returned {len(results)} results for '{query}'")
        return results

    logger.info(f"WebAgent: No results for original query '{query}', trying English keywords...")

    # ── Attempt 2: Extracted English keywords ─────────────────
    english_query = _extract_english_keywords(query)
    if english_query != query:
        time.sleep(0.5)  # Brief pause to avoid rate-limiting
        results = _do_search(english_query)
        if results:
            logger.info(f"WebAgent: DuckDuckGo returned {len(results)} results for English retry '{english_query}'")
            return results

    # ── Attempt 3: Query + "explained" ────────────────────────
    edu_query = f"{query} explained"
    time.sleep(0.5)
    results = _do_search(edu_query)
    if results:
        logger.info(f"WebAgent: DuckDuckGo returned {len(results)} results for '{edu_query}'")
        return results

    logger.warning(f"WebAgent: All DuckDuckGo attempts failed for '{query}'")
    return []


def load_web_page(url: str) -> str:
    """
    Load and extract text from a web page using LangChain's WebBaseLoader.
    Falls back to requests + BeautifulSoup if WebBaseLoader fails.

    Args:
        url: The page URL.

    Returns:
        Extracted text content, or empty string on failure.
    """
    try:
        from langchain_community.document_loaders import WebBaseLoader

        loader = WebBaseLoader(url)
        loader.requests_kwargs = {"timeout": 10}
        docs = loader.load()

        if docs:
            content = docs[0].page_content
            logger.info(f"WebAgent: Loaded {len(content)} chars from {url}")
            return content

    except Exception as e:
        logger.debug(f"WebAgent: WebBaseLoader failed for {url}: {e}")

    # Fallback: requests + BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        })
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        logger.info(f"WebAgent: Loaded {len(text)} chars from {url} (fallback)")
        return text

    except Exception as e:
        logger.error(f"WebAgent: Failed to load {url}: {e}")
        return ""


def _is_trusted(url: str) -> bool:
    """Check if a URL belongs to a trusted educational website."""
    return any(domain in url.lower() for domain in TRUSTED_WEBSITES)


def _validate_url(url: str) -> bool:
    """Check if a URL looks valid and isn't a tracking redirect."""
    if not url:
        return False
    if not url.startswith("http"):
        return False
    # Filter out known redirect/tracking URLs
    bad_patterns = ["duckduckgo.com/l/", "google.com/url?", "bing.com/ck/"]
    return not any(p in url for p in bad_patterns)


def web_search_and_rank(query: str, query_embedding: np.ndarray) -> list[dict]:
    """
    Full web search fallback pipeline:
      1. Search DuckDuckGo (with Hinglish retry)
      2. Try to load pages and chunk them
      3. If page loading fails, use DuckDuckGo snippets directly
      4. Embed, rank, and apply trusted-site boosting

    Args:
        query: The cleaned search query.
        query_embedding: The query embedding vector.

    Returns:
        List of ranked web results.
    """
    # Step 1: DuckDuckGo search (with auto-retry for Hinglish)
    search_results = search_duckduckgo(query)
    if not search_results:
        logger.warning("WebAgent: No DuckDuckGo results found after all retries.")
        return []

    # Validate URLs
    search_results = [r for r in search_results if _validate_url(r["url"])]
    if not search_results:
        logger.warning("WebAgent: No valid URLs after filtering.")
        return []

    all_chunks = []
    pages_loaded = 0

    # Step 2: Try to load, clean, chunk each URL
    for result in search_results:
        url = result["url"]

        page_text = load_web_page(url)
        if page_text:
            cleaned = clean_text(page_text)
            if len(cleaned) >= 100:
                pages_loaded += 1
                chunks = chunk_text(cleaned, TRANSCRIPT_CHUNK_SIZE, TRANSCRIPT_CHUNK_OVERLAP)
                for chunk_data in chunks:
                    all_chunks.append({
                        "title": result["title"],
                        "url": url,
                        "snippet_from_search": result["snippet"],
                        "chunk_text": chunk_data["text"],
                        "is_trusted": _is_trusted(url),
                    })
                continue  # Successfully loaded — skip to next URL

        # Page failed to load — use DuckDuckGo snippet as fallback
        snippet = result.get("snippet", "")
        if snippet and len(snippet) > 20:
            logger.info(f"WebAgent: Using search snippet for {url} (page load failed)")
            all_chunks.append({
                "title": result["title"],
                "url": url,
                "snippet_from_search": snippet,
                "chunk_text": f"{result['title']}. {snippet}",
                "is_trusted": _is_trusted(url),
            })

    logger.info(f"WebAgent: {pages_loaded}/{len(search_results)} pages loaded, "
                f"{len(all_chunks)} total chunks")

    if not all_chunks:
        logger.warning("WebAgent: No usable content from web pages or snippets.")
        return []

    # Step 3: Embed all chunks
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    chunk_embeddings = embed_texts(chunk_texts)

    if chunk_embeddings.size == 0:
        return []

    # Step 4: Compute similarity with trusted-site boosting
    similarities = compute_similarity(query_embedding, chunk_embeddings)

    for i, chunk_info in enumerate(all_chunks):
        score = similarities[i]
        if chunk_info["is_trusted"]:
            score *= TRUSTED_SITE_BOOST
        all_chunks[i]["similarity"] = float(score)

    # Step 5: Rank and deduplicate by URL (keep best chunk per URL)
    url_best = {}
    for chunk_info in all_chunks:
        url = chunk_info["url"]
        if url not in url_best or chunk_info["similarity"] > url_best[url]["similarity"]:
            url_best[url] = {
                "title": chunk_info["title"],
                "url": url,
                "snippet": chunk_info["chunk_text"][:300],
                "similarity": chunk_info["similarity"],
                "is_trusted": chunk_info["is_trusted"],
            }

    # Sort by similarity descending
    ranked = sorted(url_best.values(), key=lambda x: x["similarity"], reverse=True)

    logger.info(f"WebAgent: Returning {len(ranked)} ranked web results")
    return ranked

