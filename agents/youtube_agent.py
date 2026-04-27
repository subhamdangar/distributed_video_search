"""
YouTube Retrieval Agent
───────────────────────
3-tier timestamp resolution:

  Tier 1 — TRANSCRIPT: Load transcript → chunk → embed → match (precise timestamps)
  Tier 2 — CHAPTERS:   Fetch chapter markers from yt_dlp → embed chapter titles →
                        match query (exact chapter timestamps, works when IP blocked)
  Tier 3 — TITLE:      Match against video title only (timestamp = 00:00:00, last resort)

Each tier is tried in order. If transcripts are blocked by YouTube, the system
automatically falls back to chapter matching, which uses metadata already
available during video search.
"""

import logging
import re
import traceback
from typing import Optional
import subprocess
import tempfile
import os
import time
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from config.channels import (
    MAX_VIDEOS_PER_CHANNEL,
    TRANSCRIPT_CHUNK_SIZE,
    TRANSCRIPT_CHUNK_OVERLAP,
)
from utils.cleaning import clean_text, chunk_text
from utils.embeddings import embed_text, embed_texts

logger = logging.getLogger(__name__)

# Module-level flag: set to True when YouTube blocks our IP
_ip_blocked = False


def _compute_keyword_score(query: str, text: str) -> float:
    """
    Compute a simple keyword overlap score between the query and a text.
    Returns a value in [0, 1] based on what fraction of query words appear in the text.
    """
    if not query or not text:
        return 0.0
    query_words = set(query.lower().split())
    text_lower = text.lower()
    matches = sum(1 for w in query_words if w in text_lower)
    return matches / max(len(query_words), 1)


def _extract_videos_from_result(result: dict, channel_name: str) -> list[dict]:
    """Extract video metadata from a yt_dlp result dict."""
    videos = []
    if result is None:
        return videos

    entries = result.get("entries", []) if "entries" in result else [result]

    for entry in entries:
        if entry is None:
            continue
        video_id = entry.get("id", "")
        if not video_id or len(video_id) > 20:
            continue

        videos.append(
            {
                "video_id": video_id,
                "title": entry.get("title", "Unknown"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "view_count": entry.get("view_count", 0) or 0,
                "channel_name": channel_name,
                "duration": entry.get("duration", 0) or 0,
                "description": entry.get("description", "") or "",
            }
        )

    return videos


def fetch_channel_videos(channel: dict, search_query: str = "") -> list[dict]:
    """
    Fetch ALL videos from a YouTube channel using yt_dlp.

    NO playlistend limit — fetches complete channel metadata.
    The metadata filtering stage (process_channel Stage 1) will select
    the top-K most relevant candidates for deep processing.

    DUAL STRATEGY (both stay WITHIN the predefined channel):
      A) Search WITHIN the channel using /search?query=...
         → finds topic-specific videos even if not recent
         → this is NOT ytsearch (global search) — it's channel-scoped
      B) Fetch ALL uploads from /videos (no limit)
         → ensures full coverage of the channel

    Combined, we get complete channel coverage.
    """
    import yt_dlp
    from urllib.parse import quote_plus

    channel_name = channel["name"]
    channel_id = channel["channel_id"]
    logger.info(
        f"YouTubeAgent: Fetching ALL videos from '{channel_name}' (ID: {channel_id})"
    )

    # NO playlistend — fetch everything
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "force_generic_extractor": False,
        "ignoreerrors": True,
        # "cookiefile": "cookies.txt",
    }

    all_videos = {}

    # ── Strategy A: Search WITHIN the channel (NOT global ytsearch) ──
    # URL: https://www.youtube.com/channel/{id}/search?query=...
    # This only returns videos from THIS specific channel.
    if search_query:
        channel_search_url = (
            f"https://www.youtube.com/channel/{channel_id}"
            f"/search?query={quote_plus(search_query)}"
        )
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(channel_search_url, download=False)
                for v in _extract_videos_from_result(result, channel_name):
                    all_videos[v["video_id"]] = v

            logger.info(
                f"YouTubeAgent: Channel search found {len(all_videos)} videos "
                f"for '{channel_name}' (query='{search_query}')"
            )
        except Exception as e:
            logger.debug(
                f"YouTubeAgent: Channel search failed for '{channel_name}': {e}"
            )

    # ── Strategy B: ALL uploads from the channel (NO limit) ──
    channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(channel_url, download=False)
            count_before = len(all_videos)
            for v in _extract_videos_from_result(result, channel_name):
                if v["video_id"] not in all_videos:
                    all_videos[v["video_id"]] = v
            added = len(all_videos) - count_before
            logger.info(
                f"YouTubeAgent: Channel uploads added {added} new videos "
                f"for '{channel_name}' (total: {len(all_videos)})"
            )
    except Exception as e:
        logger.debug(f"YouTubeAgent: Channel uploads failed for '{channel_name}': {e}")

    # NO slicing — return ALL videos. Metadata filter will select top-K.
    videos = list(all_videos.values())
    logger.info(
        f"YouTubeAgent: Total {len(videos)} unique videos from '{channel_name}'"
    )
    return videos


# ═══════════════════════════════════════════════════════════════════════
# TIER 1: Transcript-based matching (best precision)
# ═══════════════════════════════════════════════════════════════════════


# def load_transcript(video_id: str, languages: list[str] = None) -> Optional[str]:
#     """Load transcript. Returns None if unavailable or IP blocked."""
#     global _ip_blocked

#     if _ip_blocked:
#         return None

#     languages = languages or ["en", "hi", "en-IN", "hi-Latn"]
#     url = f"https://www.youtube.com/watch?v={video_id}"

#     # Attempt 1: LangChain YoutubeLoader
#     try:
#         from langchain_community.document_loaders import YoutubeLoader

#         loader = YoutubeLoader.from_youtube_url(
#             url, add_video_info=False, language=languages
#         )
#         docs = loader.load()
#         if docs and docs[0].page_content.strip():
#             logger.info(f"YouTubeAgent: Transcript loaded via LangChain for {video_id}")
#             return docs[0].page_content
#     except Exception as e:
#         err_str = str(e).lower()
#         if (
#             "ipblocked" in err_str
#             or ("ip" in err_str and "block" in err_str)
#             or "429" in str(e)
#         ):
#             _ip_blocked = True
#             logger.warning(
#                 "YouTubeAgent: ⚠️  YouTube IP BLOCKED — switching to chapter-based matching."
#             )
#             return None
#         logger.debug(f"YouTubeAgent: LangChain failed for {video_id}: {e}")

#     # Attempt 2: youtube_transcript_api (v1.x)
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi

#         api = YouTubeTranscriptApi()
#         transcript_list = api.list(video_id)

#         transcript = None
#         try:
#             transcript = transcript_list.find_transcript(languages)
#         except Exception:
#             pass
#         if transcript is None:
#             try:
#                 transcript = transcript_list.find_generated_transcript(languages)
#             except Exception:
#                 pass
#         if transcript is None:
#             try:
#                 for t in transcript_list:
#                     transcript = t
#                     break
#             except Exception:
#                 pass

#         if transcript is not None:
#             fetched = transcript.fetch()
#             full_text = " ".join(
#                 (
#                     snippet.text
#                     if hasattr(snippet, "text")
#                     else str(snippet.get("text", ""))
#                 )
#                 for snippet in fetched
#             )
#             if full_text.strip():
#                 logger.info(f"YouTubeAgent: Transcript loaded via API for {video_id}")
#                 return full_text

#     except Exception as e:
#         err_str = str(e).lower()
#         if (
#             "ipblocked" in err_str
#             or ("ip" in err_str and "block" in err_str)
#             or "429" in str(e)
#         ):
#             _ip_blocked = True
#             logger.warning(
#                 "YouTubeAgent: ⚠️  YouTube IP BLOCKED — switching to chapter-based matching."
#             )
#             return None
#         logger.debug(f"YouTubeAgent: API failed for {video_id}: {e}")

#     return None




# ===================================================================================
#              BELOW PART IS MODIFIED
# ===================================================================================






# def load_transcript(video_id: str, languages: list[str] = None) -> Optional[str]:
#     """Load transcript with robust multi-method fallback."""
#     global _ip_blocked

#     if _ip_blocked:
#         return None

#     languages = languages or ["en", "hi", "en-IN", "hi-Latn"]
#     url = f"https://www.youtube.com/watch?v={video_id}"

#     # ─────────────────────────────────────────────
#     # Attempt 0: yt-dlp subtitles (NEW - strongest)
#     # ─────────────────────────────────────────────
#     try:
#         with tempfile.TemporaryDirectory() as tmpdir:
#             output_template = os.path.join(tmpdir, "sub")

#             command = [
#                 "yt-dlp",
#                 "--skip-download",
#                 "--write-auto-sub",
#                 "--write-sub",
#                 "--sub-lang", "en",
#                 "--sub-format", "vtt",
#                 "--no-warnings",
#                 "--ignore-errors",
#                 "--format", "skip",
#                 "-o", output_template,
#                 url
#             ]

#             subprocess.run(command, capture_output=True)

#             # Find VTT file
#             for file in os.listdir(tmpdir):
#                 if file.endswith(".vtt"):
#                     path = os.path.join(tmpdir, file)

#                     with open(path, "r", encoding="utf-8") as f:
#                         text = f.read()

#                     # Clean VTT → plain text
#                     text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d+ --> .*', '', text)
#                     text = re.sub(r'<.*?>', '', text)
#                     text = re.sub(r'\n+', ' ', text)

#                     if text.strip():
#                         logger.info(f"YouTubeAgent: Transcript loaded via yt-dlp for {video_id}")
#                         return text.strip()

#     except Exception as e:
#         logger.debug(f"YouTubeAgent: yt-dlp failed for {video_id}: {e}")

#     # ─────────────────────────────────────────────
#     # Attempt 1: LangChain YoutubeLoader (existing)
#     # ─────────────────────────────────────────────
#     try:
#         from langchain_community.document_loaders import YoutubeLoader

#         loader = YoutubeLoader.from_youtube_url(
#             url, add_video_info=False, language=languages
#         )
#         docs = loader.load()
#         if docs and docs[0].page_content.strip():
#             logger.info(f"YouTubeAgent: Transcript loaded via LangChain for {video_id}")
#             return docs[0].page_content

#     except Exception as e:
#         err_str = str(e).lower()
#         if (
#             "ipblocked" in err_str
#             or ("ip" in err_str and "block" in err_str)
#             or "429" in str(e)
#         ):
#             _ip_blocked = True
#             logger.warning(
#                 "YouTubeAgent: ⚠️  YouTube IP BLOCKED — switching to chapter-based matching."
#             )
#             return None

#         logger.debug(f"YouTubeAgent: LangChain failed for {video_id}: {e}")

#     # ─────────────────────────────────────────────
#     # Attempt 2: youtube_transcript_api (existing)
#     # ─────────────────────────────────────────────
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi

#         api = YouTubeTranscriptApi()
#         transcript_list = api.list(video_id)

#         transcript = None
#         try:
#             transcript = transcript_list.find_transcript(languages)
#         except Exception:
#             pass
#         if transcript is None:
#             try:
#                 transcript = transcript_list.find_generated_transcript(languages)
#             except Exception:
#                 pass
#         if transcript is None:
#             try:
#                 for t in transcript_list:
#                     transcript = t
#                     break
#             except Exception:
#                 pass

#         if transcript is not None:
#             fetched = transcript.fetch()
#             full_text = " ".join(
#                 snippet.text if hasattr(snippet, "text")
#                 else str(snippet.get("text", ""))
#                 for snippet in fetched
#             )

#             if full_text.strip():
#                 logger.info(f"YouTubeAgent: Transcript loaded via API for {video_id}")
#                 return full_text

#     except Exception as e:
#         err_str = str(e).lower()
#         if (
#             "ipblocked" in err_str
#             or ("ip" in err_str and "block" in err_str)
#             or "429" in str(e)
#         ):
#             _ip_blocked = True
#             logger.warning(
#                 "YouTubeAgent: ⚠️  YouTube IP BLOCKED — switching to chapter-based matching."
#             )
#             return None

#         logger.debug(f"YouTubeAgent: API failed for {video_id}: {e}")

#     return None

# def process_video_with_transcript(
#     video: dict, query_embedding: np.ndarray
# ) -> list[dict]:
#     """TIER 1: Transcript-based matching. Returns chunk results with precise timestamps."""
#     from utils.similarity import compute_similarity

#     transcript = load_transcript(video["video_id"])
#     if not transcript:
#         return []

#     transcript = clean_text(transcript)
#     if len(transcript) < 50:
#         return []

#     chunks = chunk_text(transcript, TRANSCRIPT_CHUNK_SIZE, TRANSCRIPT_CHUNK_OVERLAP)
#     if not chunks:
#         return []

#     chunk_texts = [c["text"] for c in chunks]
#     chunk_embeddings = embed_texts(chunk_texts)

#     if chunk_embeddings.size == 0:
#         return []

#     similarities = compute_similarity(query_embedding, chunk_embeddings)
#     duration = video.get("duration", 0)
#     total_chars = sum(len(c["text"]) for c in chunks)

#     # ── FILTER: discard chunks below minimum similarity ──
#     MIN_CHUNK_SIMILARITY = 0.30

#     results = []
#     for i, chunk_data in enumerate(chunks):
#         sim = float(similarities[i])
#         if sim < MIN_CHUNK_SIMILARITY:
#             continue  # Skip weak chunks early

#         if duration > 0 and total_chars > 0:
#             timestamp_sec = int((chunk_data["start_char"] / total_chars) * duration)
#         else:
#             timestamp_sec = 0

#         timestamp_str = f"{timestamp_sec // 3600:02d}:{(timestamp_sec % 3600) // 60:02d}:{timestamp_sec % 60:02d}"

#         results.append(
#             {
#                 "video_id": video["video_id"],
#                 "title": video["title"],
#                 "url": video["url"],
#                 "channel_name": video["channel_name"],
#                 "view_count": video["view_count"],
#                 "duration": duration,
#                 "chunk_text": chunk_data["text"][:200],
#                 "chunk_index": chunk_data["chunk_index"],
#                 "timestamp_sec": timestamp_sec,
#                 "timestamp_str": timestamp_str,
#                 "timestamp_url": f"{video['url']}&t={timestamp_sec}s",
#                 "similarity": sim,
#                 "match_type": "transcript",
#             }
#         )

#     return results


# ═══════════════════════════════════════════════════════════════════════
# TIER 2: Chapter-based matching (good precision, works when IP blocked)
# ═══════════════════════════════════════════════════════════════════════


class _SilentLogger:
    """Suppress all yt_dlp log/error output (it ignores logger=None)."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass




def fetch_video_chapters(video_id: str) -> list[dict]:
    """
    Fetch chapter markers for a video using yt_dlp's full metadata extraction.
    Chapters are embedded in YouTube video metadata and don't require
    transcript access, so they work even when YouTube blocks the IP.

    Returns list of dicts: [{title, start_time, end_time}, ...]
    """
    import yt_dlp
    import sys
    import io

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "ignore_no_formats_error": True,
        "logger": _SilentLogger(),
        "logtostderr": False,
        # "cookiefile": "cookies.txt",
    }

    old_stderr = sys.stderr
    try:
        # Redirect stderr to suppress yt_dlp's direct stderr writes
        sys.stderr = io.StringIO()
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False
                )
        finally:
            sys.stderr = old_stderr

        if info is None:
            return []

        chapters = info.get("chapters", [])

        # If no chapters, try parsing timestamps from description
        if not chapters:
            chapters = _parse_description_timestamps(
                info.get("description", ""), info.get("duration", 0)
            )

        return chapters

    except Exception as e:
        sys.stderr = old_stderr  # Ensure stderr is restored on exception
        logger.debug(f"YouTubeAgent: Chapter fetch failed for {video_id}: {e}")
        return []


def _parse_description_timestamps(description: str, duration: int) -> list[dict]:
    """
    Parse timestamps from a YouTube video description.
    Many educational videos list topics with timestamps like:
      0:00 Introduction
      5:32 What are Operators
      12:45 Arithmetic Operators

    Returns list of chapter-like dicts.
    """
    if not description:
        return []

    # Match patterns like "0:00", "00:00", "1:23:45", "01:23:45"
    # Greedy .+ is required — lazy .+? skips alternating lines on consecutive timestamps.
    pattern = r"(?:^|\n)\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—: ]\s*(.+)"
    matches = re.findall(pattern, description)

    if len(matches) < 2:
        return []

    chapters = []
    for time_str, title in matches:
        parts = time_str.split(":")
        if len(parts) == 2:
            secs = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            continue

        chapters.append(
            {
                "title": title.strip(),
                "start_time": secs,
            }
        )

    # Add end times
    for i in range(len(chapters) - 1):
        chapters[i]["end_time"] = chapters[i + 1]["start_time"]
    if chapters:
        chapters[-1]["end_time"] = (
            duration if duration > 0 else chapters[-1]["start_time"] + 300
        )

    return chapters


def process_video_with_chapters(video: dict, query_embedding: np.ndarray) -> list[dict]:
    """
    TIER 2: Chapter-based matching.
    Fetches video chapters, embeds chapter titles, and matches against query.
    Returns results with accurate chapter timestamps.
    """
    from utils.similarity import compute_similarity

    video_id = video["video_id"]

    # Try parsing chapters from description snippet (already available, zero cost)
    description = video.get("description", "")
    duration = video.get("duration", 0)
    chapters = _parse_description_timestamps(description, duration) if description else []

    # Fall back to full metadata extraction via yt_dlp (no API key needed)
    if not chapters:
        chapters = fetch_video_chapters(video_id)

    if not chapters:
        return []

    logger.info(f"YouTubeAgent: Found {len(chapters)} chapters for {video_id}")

    # Embed chapter titles
    chapter_titles = [ch.get("title", "") for ch in chapters]
    chapter_titles = [t for t in chapter_titles if t.strip()]
    if not chapter_titles:
        return []

    chapter_embeddings = embed_texts(chapter_titles)
    if chapter_embeddings.size == 0:
        return []

    similarities = compute_similarity(query_embedding, chapter_embeddings)

    # ── FILTER: discard weak chapter matches ──
    MIN_CHAPTER_SIMILARITY = 0.30

    results = []
    for i, ch in enumerate(chapters):
        if i >= len(similarities):
            break

        sim = float(similarities[i])
        if sim < MIN_CHAPTER_SIMILARITY:
            continue  # Skip weak chapter matches early

        title = ch.get("title", "")
        if not title.strip():
            continue

        timestamp_sec = int(ch.get("start_time", 0))
        timestamp_str = f"{timestamp_sec // 3600:02d}:{(timestamp_sec % 3600) // 60:02d}:{timestamp_sec % 60:02d}"

        results.append(
            {
                "video_id": video_id,
                "title": video["title"],
                "url": video["url"],
                "channel_name": video["channel_name"],
                "view_count": video["view_count"],
                "duration": video.get("duration", 0),
                "chunk_text": f"[Chapter: {title}] in \"{video['title']}\"",
                "chunk_index": i,
                "timestamp_sec": timestamp_sec,
                "timestamp_str": timestamp_str,
                "timestamp_url": f"{video['url']}&t={timestamp_sec}s",
                "similarity": sim,
                "match_type": "chapter",
            }
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# TIER 3: Title-based matching (last resort, no timestamps)
# ═══════════════════════════════════════════════════════════════════════


def process_video_title_only(video: dict, query_embedding: np.ndarray) -> dict:
    """TIER 3: Match query against video title + description. Timestamp = 00:00:00."""
    from utils.similarity import compute_similarity

    match_text = video["title"]
    desc = video.get("description", "")
    if desc:
        match_text += " " + desc[:300]

    title_embedding = embed_text(match_text)
    similarity = compute_similarity(query_embedding, title_embedding.reshape(1, -1))

    return {
        "video_id": video["video_id"],
        "title": video["title"],
        "url": video["url"],
        "channel_name": video["channel_name"],
        "view_count": video["view_count"],
        "duration": video.get("duration", 0),
        "chunk_text": f"[Matched by title] {video['title']}",
        "chunk_index": 0,
        "timestamp_sec": 0,
        "timestamp_str": "00:00:00",
        "timestamp_url": video["url"],
        "similarity": float(similarity[0]),
        "match_type": "title",
    }



def process_channel(
    channel: dict, search_query: str, query_embedding: np.ndarray
) -> list[dict]:
    """
    Process an entire channel with 2-stage retrieval + 3-tier fallback.
    """

    # ─────────────────────────────────────────────────────────────
    # 🔴 WORKER DEBUG BLOCK (proof of execution on worker machine)
    # ─────────────────────────────────────────────────────────────
    import socket
    import os
    import time

    try:
        from dask.distributed import get_worker

        worker = get_worker()
        worker_id = worker.address
    except Exception:
        worker_id = "LOCAL (not distributed)"

    hostname = socket.gethostname()
    pid = os.getpid()

    print(
        f"""
================= WORKER EXECUTION =================
Host        : {hostname}
PID         : {pid}
Worker ID   : {worker_id}
Channel     : {channel['name']}
===================================================
"""
    )

    start_time = time.time()

    # ─────────────────────────────────────────────────────────────
    logger.info(f"YouTubeAgent: Processing channel '{channel['name']}'")

    videos = fetch_channel_videos(channel, search_query)
    if not videos:
        logger.info(f"YouTubeAgent: No videos found for '{channel['name']}'")
        return []

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: METADATA FILTERING
    # ─────────────────────────────────────────────────────────────
    from utils.embeddings import embed_texts
    from utils.similarity import compute_similarity

    print(
        f"[WORKER] ({hostname}) → Stage 1 START for {channel['name']} | Total videos: {len(videos)}"
    )

    video_texts = [v["title"] + " " + v.get("description", "")[:200] for v in videos]

    video_embeddings = embed_texts(video_texts)

    if video_embeddings.size == 0:
        return []

    similarities = compute_similarity(query_embedding, video_embeddings)

    for i, v in enumerate(videos):
        v["quick_score"] = float(similarities[i])
        v["keyword_score"] = _compute_keyword_score(
            search_query, v["title"] + " " + v.get("description", "")[:200]
        )

    videos = sorted(videos, key=lambda x: x["quick_score"], reverse=True)

    MIN_QUICK_SCORE = 0.25
    videos = [v for v in videos if v["quick_score"] >= MIN_QUICK_SCORE]

    if not videos:
        logger.info(
            f"YouTubeAgent: No videos passed metadata filter for '{channel['name']}'"
        )
        return []

    TOP_K = 20
    videos = videos[:TOP_K]

    print(
        f"[WORKER] ({hostname}) → Stage 1 DONE for {channel['name']} | Selected: {len(videos)}"
    )

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: DEEP PROCESSING
    # ─────────────────────────────────────────────────────────────
    print(f"[WORKER] ({hostname}) → Stage 2 START for {channel['name']}")

    all_results = []
    # tier1_count = 0
    tier2_count = 0
    tier3_count = 0

    for video in videos:
        time.sleep(random.uniform(0.3, 0.6))  # Small delay to reduce YouTube rate-limiting
        try:
            # TIER 1
            # results = process_video_with_transcript(video, query_embedding)
            # if results:
            #     tier1_count += 1
            #     for r in results:
            #         r["title_similarity"] = video["quick_score"]
            #         r["keyword_score"] = video["keyword_score"]
            #     all_results.extend(results)
            #     continue

            # TIER 2
            results = process_video_with_chapters(video, query_embedding)
            if results:
                tier2_count += 1
                for r in results:
                    r["title_similarity"] = video["quick_score"]
                    r["keyword_score"] = video["keyword_score"]
                all_results.extend(results)
                continue

            # TIER 3
            result = process_video_title_only(video, query_embedding)
            if result["similarity"] > 0.35:
                tier3_count += 1
                result["title_similarity"] = video["quick_score"]
                result["keyword_score"] = video["keyword_score"]
                all_results.append(result)

        except Exception as e:
            logger.error(
                f"YouTubeAgent: Error processing '{video.get('title', '?')}': {e}"
            )
            continue

    # ─────────────────────────────────────────────────────────────
    # FINAL SUMMARY (worker-side)
    # ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

#     print(
#         f"""
# [WORKER COMPLETE]
# Host        : {hostname}
# Channel     : {channel['name']}
# T1 (transcript) : {tier1_count}
# T2 (chapters)   : {tier2_count}
# T3 (title)      : {tier3_count}
# Total       : {len(all_results)}
# Time        : {elapsed:.2f}s
# ---------------------------------------------------
# """
#     )


    print(
        f"""
    [WORKER COMPLETE]
    Host        : {hostname}
    Channel     : {channel['name']}
    T1 (chapters)   : {tier2_count}
    T2 (title)      : {tier3_count}
    Total       : {len(all_results)}
    Time        : {elapsed:.2f}s
    ---------------------------------------------------
    """
        )

    return all_results
