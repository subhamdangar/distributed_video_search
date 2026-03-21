"""
Ranking Agent
─────────────
Implements the two-step ranking logic:

  Step 1: Select Top 10 results by semantic similarity
  Step 2: From those, select Top 3 by view_count (popularity)

This ensures results are both relevant AND popular.
Also handles deduplication (same video, different chunks → keep best chunk).
"""

import logging
from config.channels import TOP_SIMILARITY_RESULTS, TOP_FINAL_RESULTS

logger = logging.getLogger(__name__)


class RankingAgent:
    """
    Ranks YouTube chunk results using a two-stage process:
      1. Semantic relevance (cosine similarity)
      2. Popularity (view count)
    """

    def rank(self, results: list[dict],
             top_similarity: int = None,
             top_final: int = None) -> list[dict]:
        """
        Two-step ranking pipeline.

        Args:
            results: List of chunk result dicts from YouTube agent.
                     Each must have 'similarity', 'view_count', 'video_id'.
            top_similarity: Number of results to keep after similarity ranking.
            top_final: Number of final results to return.

        Returns:
            List of top-ranked results.
        """
        top_similarity = top_similarity or TOP_SIMILARITY_RESULTS
        top_final = top_final or TOP_FINAL_RESULTS

        if not results:
            logger.info("RankingAgent: No results to rank.")
            return []

        logger.info(f"RankingAgent: Ranking {len(results)} total chunks")

        # ── Deduplication: keep best chunk per video ──────────
        video_best = {}
        for r in results:
            vid = r["video_id"]
            if vid not in video_best or r["similarity"] > video_best[vid]["similarity"]:
                video_best[vid] = r

        deduplicated = list(video_best.values())
        logger.info(f"RankingAgent: {len(deduplicated)} unique videos after dedup")

        # ── Step 1: Top N by similarity ───────────────────────
        sorted_by_sim = sorted(deduplicated, key=lambda x: x["similarity"], reverse=True)
        top_by_sim = sorted_by_sim[:top_similarity]

        logger.info(f"RankingAgent: Top {len(top_by_sim)} by similarity — "
                     f"scores: {[f'{r['similarity']:.4f}' for r in top_by_sim[:5]]}")

        # ── Step 2: Top K by view_count ───────────────────────
        sorted_by_views = sorted(top_by_sim, key=lambda x: x.get("view_count", 0), reverse=True)
        final = sorted_by_views[:top_final]

        logger.info(f"RankingAgent: Final {len(final)} results — "
                     f"views: {[r.get('view_count', 0) for r in final]}")

        # ── Add rank metadata ─────────────────────────────────
        for i, r in enumerate(final):
            r["rank"] = i + 1

        return final

    def has_relevant_results(self, results: list[dict],
                              min_similarity: float = 0.25) -> bool:
        """
        Check whether any results meet the minimum relevance threshold.
        Used by the orchestrator to decide if web fallback is needed.

        Args:
            results: Ranked results.
            min_similarity: Minimum similarity score to consider relevant.

        Returns:
            True if at least one result is above the threshold.
        """
        if not results:
            return False

        best = max(r["similarity"] for r in results)
        relevant = best >= min_similarity

        logger.info(f"RankingAgent: Best similarity = {best:.4f}, "
                     f"threshold = {min_similarity}, relevant = {relevant}")
        return relevant
