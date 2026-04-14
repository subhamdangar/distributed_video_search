# """
# Ranking Agent
# ─────────────
# Implements the two-step ranking logic:

#   Step 1: Select Top 10 results by semantic similarity
#   Step 2: From those, select Top 3 by view_count (popularity)

# This ensures results are both relevant AND popular.
# Also handles deduplication (same video, different chunks → keep best chunk).
# """

# import logging
# from config.channels import TOP_SIMILARITY_RESULTS, TOP_FINAL_RESULTS

# logger = logging.getLogger(__name__)


# class RankingAgent:
#     """
#     Ranks YouTube chunk results using a two-stage process:
#       1. Semantic relevance (cosine similarity)
#       2. Popularity (view count)
#     """

#     def rank(self, results: list[dict],
#              top_similarity: int = None,
#              top_final: int = None) -> list[dict]:
#         """
#         Two-step ranking pipeline.

#         Args:
#             results: List of chunk result dicts from YouTube agent.
#                      Each must have 'similarity', 'view_count', 'video_id'.
#             top_similarity: Number of results to keep after similarity ranking.
#             top_final: Number of final results to return.

#         Returns:
#             List of top-ranked results.
#         """
#         top_similarity = top_similarity or TOP_SIMILARITY_RESULTS
#         top_final = top_final or TOP_FINAL_RESULTS

#         if not results:
#             logger.info("RankingAgent: No results to rank.")
#             return []

#         logger.info(f"RankingAgent: Ranking {len(results)} total chunks")

#         # ── Deduplication: keep best chunk per video ──────────
#         video_best = {}
#         for r in results:
#             vid = r["video_id"]
#             if vid not in video_best or r["similarity"] > video_best[vid]["similarity"]:
#                 video_best[vid] = r

#         deduplicated = list(video_best.values())
#         logger.info(f"RankingAgent: {len(deduplicated)} unique videos after dedup")

#         # ── Step 1: Top N by similarity ───────────────────────
#         sorted_by_sim = sorted(deduplicated, key=lambda x: x["similarity"], reverse=True)
#         top_by_sim = sorted_by_sim[:top_similarity]

#         logger.info(f"RankingAgent: Top {len(top_by_sim)} by similarity — "
#                      f"scores: {[f'{r['similarity']:.4f}' for r in top_by_sim[:5]]}")

#         # ── Step 2: Top K by view_count ───────────────────────
#         sorted_by_views = sorted(top_by_sim, key=lambda x: x.get("view_count", 0), reverse=True)
#         final = sorted_by_views[:top_final]

#         logger.info(f"RankingAgent: Final {len(final)} results — "
#                      f"views: {[r.get('view_count', 0) for r in final]}")

#         # ── Add rank metadata ─────────────────────────────────
#         for i, r in enumerate(final):
#             r["rank"] = i + 1

#         return final

#     def has_relevant_results(self, results: list[dict],
#                               min_similarity: float = 0.30) -> bool:
#         """
#         Check whether any results meet the minimum relevance threshold.
#         Used by the orchestrator to decide if web fallback is needed.

#         Args:
#             results: Ranked results.
#             min_similarity: Minimum similarity score to consider relevant.

#         Returns:
#             True if at least one result is above the threshold.
#         """
#         if not results:
#             return False

#         best = max(r["similarity"] for r in results)
#         relevant = best >= min_similarity

#         logger.info(f"RankingAgent: Best similarity = {best:.4f}, "
#                      f"threshold = {min_similarity}, relevant = {relevant}")
#         return relevant





"""
Ranking Agent
─────────────
Implements a multi-signal ranking pipeline with strong filtering.

DESIGN:

  Step 0: Deduplicate (same video → keep best chunk)
  Step 1: Compute hybrid final_score using 3 signals:
          = 0.50 * chunk_similarity     (transcript/chapter deep match)
          + 0.30 * title_similarity     (metadata intent match)
          + 0.20 * keyword_score        (keyword overlap)
  Step 2: Select Top N by final_score
  Step 3: Re-sort Top N by final_score (primary) + view_count (tiebreaker)
          → views NO LONGER override relevance
  Step 4: Return Top K

Why 3-signal scoring?
- chunk_similarity → deep semantic match (transcript/chapter content)
- title_similarity → intent match (does the video topic match the query?)
- keyword_score   → exact keyword overlap (catches literal matches)
- view_count      → tiebreaker only (prevents viral-but-irrelevant videos
                     from dominating)

This ensures results are:
✔ Deeply relevant (semantic match in content)
✔ Intent-aware (title/metadata match)
✔ Keyword-verified (literal terms present)
✔ Popular (as tiebreaker, not primary signal)
"""

import logging
from config.channels import TOP_SIMILARITY_RESULTS, TOP_FINAL_RESULTS

logger = logging.getLogger(__name__)


class RankingAgent:
    """
    Ranks YouTube results using 3-signal hybrid scoring.
    Views are used as a tiebreaker, NOT the primary ranking signal.
    """

    # ── Scoring weights (tune these to adjust relevance vs popularity) ──
    W_CHUNK = 0.50       # weight for transcript/chapter chunk similarity
    W_TITLE = 0.30       # weight for title/metadata similarity
    W_KEYWORD = 0.20     # weight for keyword overlap score

    def rank(self, results: list[dict],
             top_similarity: int = None,
             top_final: int = None) -> list[dict]:
        """
        Multi-signal ranking pipeline.

        Args:
            results: List of chunk-level results from YouTube agent.
                     Each result should contain:
                       - similarity (float)       — from Tier 1/2/3
                       - title_similarity (float)  — from metadata stage
                       - keyword_score (float)     — from keyword matching
                       - video_id (str)
                       - view_count (int)

            top_similarity: Number of results to keep after scoring
            top_final: Number of final results

        Returns:
            List of top-ranked results with 'final_score' and 'rank' added.
        """

        top_similarity = top_similarity or TOP_SIMILARITY_RESULTS
        top_final = top_final or TOP_FINAL_RESULTS

        if not results:
            logger.info("RankingAgent: No results to rank.")
            return []

        logger.info(f"RankingAgent: Ranking {len(results)} total chunks")

        # ──────────────────────────────────────────────────────
        # STEP 0: Deduplicate (keep best chunk per video)
        # ──────────────────────────────────────────────────────
        video_best = {}

        for r in results:
            vid = r["video_id"]

            if vid not in video_best or r["similarity"] > video_best[vid]["similarity"]:
                video_best[vid] = r

        deduplicated = list(video_best.values())

        logger.info(f"RankingAgent: {len(deduplicated)} unique videos after dedup")

        # ──────────────────────────────────────────────────────
        # STEP 1: 3-signal hybrid scoring
        # ──────────────────────────────────────────────────────
        for r in deduplicated:
            chunk_score = r.get("similarity", 0.0)
            title_score = r.get("title_similarity", 0.0)
            kw_score = r.get("keyword_score", 0.0)

            r["final_score"] = (
                self.W_CHUNK * chunk_score +
                self.W_TITLE * title_score +
                self.W_KEYWORD * kw_score
            )

        # ──────────────────────────────────────────────────────
        # STEP 2: Sort by final_score (primary), view_count (tiebreaker)
        # ──────────────────────────────────────────────────────
        sorted_results = sorted(
            deduplicated,
            key=lambda x: (x["final_score"], x.get("view_count", 0)),
            reverse=True
        )

        top_by_score = sorted_results[:top_similarity]

        # Log detailed breakdown for top results
        for r in top_by_score[:5]:
            logger.info(
                f"  → {r.get('title', '?')[:50]} | "
                f"final={r['final_score']:.4f} "
                f"(chunk={r.get('similarity', 0):.3f}, "
                f"title={r.get('title_similarity', 0):.3f}, "
                f"kw={r.get('keyword_score', 0):.2f}) | "
                f"views={r.get('view_count', 0):,}"
            )

        # ──────────────────────────────────────────────────────
        # STEP 3: Take final Top K (already sorted by relevance)
        # ──────────────────────────────────────────────────────
        final = top_by_score[:top_final]

        logger.info(
            f"RankingAgent: Final {len(final)} results — "
            f"scores: {[round(r['final_score'], 4) for r in final]}"
        )

        # ──────────────────────────────────────────────────────
        # STEP 4: Add rank metadata
        # ──────────────────────────────────────────────────────
        for i, r in enumerate(final):
            r["rank"] = i + 1

        return final

    # ──────────────────────────────────────────────────────────
    # FALLBACK DECISION (YouTube vs Web)
    # ──────────────────────────────────────────────────────────
    def has_relevant_results(self, results: list[dict],
                            min_similarity: float = 0.45) -> bool:
        """
        Decide whether YouTube results are good enough to return.

        Uses hybrid final_score (not raw similarity) as the decision criteria.
        Threshold is set to 0.45 — stricter than before (was 0.30) to avoid
        returning weakly-matched results.

        Args:
            results: Ranked results
            min_similarity: minimum final_score threshold

        Returns:
            True → use YouTube results
            False → fallback to web search
        """

        if not results:
            return False

        # Use hybrid score for decision
        best = max(r.get("final_score", r.get("similarity", 0)) for r in results)

        relevant = best >= min_similarity

        logger.info(
            f"RankingAgent: Best final_score = {best:.4f}, "
            f"threshold = {min_similarity}, relevant = {relevant}"
        )

        return relevant