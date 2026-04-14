"""
Orchestrator Agent
──────────────────
The master controller that runs the entire pipeline:

  1. Query Understanding → clean + embed
  2. Cache Check → return if hit
  3. Route to Subject(s)
  4. Dask Parallel Execution → process channels
  5. Rank Results
  6. Fallback to Web if needed
  7. Store in Cache
  8. Return formatted output

Uses Dask's `delayed` + `compute` for parallelism across channels.
"""

import logging
import time
import dask
from dask import delayed, compute

from config.channels import SUBJECT_CHANNELS, MAX_DASK_WORKERS
from agents.query_agent import QueryAgent
from agents.cache_agent import CacheAgent
from agents.router_agent import RouterAgent
from agents.youtube_agent import process_channel
from agents.ranking_agent import RankingAgent
from agents.web_agent import web_search_and_rank

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Controls the full search pipeline with distributed execution via Dask.

    In distributed mode (DASK_SCHEDULER_ADDRESS is set):
      - Client connects to the scheduler once on __init__
      - Tasks are submitted to remote workers via client.submit()
      - Workers execute process_channel() on different machines
      - Results are gathered back to the client

    In local mode (DASK_SCHEDULER_ADDRESS is None):
      - Falls back to local threaded scheduler
    """

    def __init__(self):
        self.query_agent = QueryAgent()
        self.cache_agent = CacheAgent()
        self.router_agent = RouterAgent()
        self.ranking_agent = RankingAgent()

        # ── Dask Distributed Client (persistent connection) ──
        from config.channels import DASK_SCHEDULER_ADDRESS
        self._dask_client = None
        self._distributed_mode = False

        if DASK_SCHEDULER_ADDRESS:
            try:
                from dask.distributed import Client
                self._dask_client = Client(DASK_SCHEDULER_ADDRESS)
                workers = self._dask_client.scheduler_info()["workers"]
                self._distributed_mode = True
                logger.info(
                    f"Orchestrator: Connected to Dask scheduler at {DASK_SCHEDULER_ADDRESS} "
                    f"| Workers: {len(workers)} "
                    f"| Worker addresses: {list(workers.keys())}"
                )
            except Exception as e:
                logger.warning(f"Orchestrator: Failed to connect to Dask scheduler: {e}")
                logger.warning("Orchestrator: Falling back to local threaded execution.")
                self._distributed_mode = False
        else:
            logger.info("Orchestrator: Running in LOCAL mode (no scheduler address set).")

        logger.info("Orchestrator: All agents initialized.")

    def search(self, raw_query: str) -> dict:
        """
        Execute the full search pipeline.

        Args:
            raw_query: The user's natural-language query.

        Returns:
            Result dict with keys:
              - 'source': 'youtube' | 'web' | 'cache'
              - 'query': cleaned query
              - 'results': list of result items
              - 'execution_time': time in seconds
              - 'scores': optional evaluation scores
        """
        start_time = time.time()

        # ══════════════════════════════════════════════════════
        # Stage 1: Query Understanding
        # ══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("STAGE 1: Query Understanding")
        logger.info("=" * 60)

        query_data = self.query_agent.process(raw_query)
        cleaned_query = query_data["cleaned"]
        query_embedding = query_data["embedding"]

        if not cleaned_query:
            return self._error_result("Empty query after cleaning.", start_time)

        # ══════════════════════════════════════════════════════
        # Stage 2: Cache Check
        # ══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("STAGE 2: Semantic Cache Check")
        logger.info("=" * 60)

        cache_hit = self.cache_agent.search(query_embedding)
        if cache_hit:
            elapsed = time.time() - start_time
            logger.info(f"Cache hit! Returning cached results. ({elapsed:.2f}s)")
            return {
                "source": "cache",
                "query": cleaned_query,
                "cached_query": cache_hit["cached_query"],
                "cache_similarity": cache_hit["similarity"],
                "results": cache_hit["results"],
                "execution_time": elapsed,
            }

        # ══════════════════════════════════════════════════════
        # Stage 3: Topic Routing
        # ══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("STAGE 3: Topic Routing")
        logger.info("=" * 60)

        subjects = self.router_agent.route(cleaned_query, query_embedding)
        logger.info(f"Routed to subjects: {subjects}")

        # Gather all channels to process
        channels_to_process = []
        for subject in subjects:
            channels = SUBJECT_CHANNELS.get(subject, [])
            channels_to_process.extend(channels)

        if not channels_to_process:
            logger.warning("No channels found for routed subjects.")
            return self._web_fallback(cleaned_query, query_embedding, start_time)

        # Remove duplicate channels (same channel_id in multiple subjects)
        seen_ids = set()
        unique_channels = []
        for ch in channels_to_process:
            if ch["channel_id"] not in seen_ids:
                seen_ids.add(ch["channel_id"])
                unique_channels.append(ch)

        logger.info(f"Processing {len(unique_channels)} unique channels: "
                     f"{[c['name'] for c in unique_channels]}")

        # ══════════════════════════════════════════════════════
        # Stage 4: Dask Parallel Channel Processing
        # ══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("STAGE 4: Dask Distributed Execution")
        logger.info("=" * 60)

        # ── Execute tasks ─────────────────────────────────────
        try:
            if self._distributed_mode and self._dask_client:
                # ═══ DISTRIBUTED MODE ═══
                # Submit each channel as a task to the Dask scheduler.
                # The scheduler distributes tasks to worker machines.
                # client.submit() → Future, client.gather() → results
                client = self._dask_client

                workers = client.scheduler_info()["workers"]
                logger.info(
                    f"Submitting {len(unique_channels)} tasks to "
                    f"{len(workers)} distributed workers"
                )

                futures = []
                for channel in unique_channels:
                    future = client.submit(
                        process_channel, channel, cleaned_query, query_embedding
                    )
                    futures.append(future)

                # Gather results from all workers (blocks until done)
                channel_results = client.gather(futures)

                logger.info(
                    f"All {len(channel_results)} tasks completed on workers."
                )
            else:
                # ═══ LOCAL MODE ═══
                # Use Dask delayed + threaded scheduler on this machine
                delayed_tasks = []
                for channel in unique_channels:
                    task = delayed(process_channel)(channel, cleaned_query, query_embedding)
                    delayed_tasks.append(task)

                logger.info(f"Created {len(delayed_tasks)} Dask delayed tasks (local).")

                channel_results = compute(
                    *delayed_tasks,
                    scheduler="threads",
                    num_workers=min(MAX_DASK_WORKERS, len(delayed_tasks)),
                )
        except Exception as e:
            logger.error(f"Dask execution error: {e}")
            # Fallback: execute sequentially
            channel_results = []
            for channel in unique_channels:
                try:
                    result = process_channel(channel, cleaned_query, query_embedding)
                    channel_results.append(result)
                except Exception as ex:
                    logger.error(f"Sequential fallback error for {channel['name']}: {ex}")
                    channel_results.append([])

        # Flatten results from all channels
        all_results = []
        for channel_result in channel_results:
            if channel_result:
                all_results.extend(channel_result)

        logger.info(f"Total chunks from all channels: {len(all_results)}")

        # ══════════════════════════════════════════════════════
        # Stage 5: Ranking
        # ══════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("STAGE 5: Ranking")
        logger.info("=" * 60)

        ranked = self.ranking_agent.rank(all_results)

        # ══════════════════════════════════════════════════════
        # Stage 6: Check Relevance → Fallback if needed
        # ══════════════════════════════════════════════════════
        if not self.ranking_agent.has_relevant_results(ranked):
            logger.info("No relevant YouTube results. Falling back to web search.")
            return self._web_fallback(cleaned_query, query_embedding, start_time)

        # ══════════════════════════════════════════════════════
        # Stage 7: Format + Cache + Return
        # ══════════════════════════════════════════════════════
        formatted = self._format_youtube_results(ranked)
        elapsed = time.time() - start_time

        result = {
            "source": "youtube",
            "query": cleaned_query,
            "results": formatted,
            "total_chunks_processed": len(all_results),
            "channels_searched": len(unique_channels),
            "execution_time": elapsed,
            "scores": self._evaluation_scores(ranked),
        }

        # Store in cache
        self.cache_agent.store(cleaned_query, query_embedding, formatted)

        logger.info(f"Pipeline complete. ({elapsed:.2f}s)")
        return result

    def _web_fallback(self, query: str, query_embedding, start_time: float) -> dict:
        """Execute web search fallback pipeline."""
        logger.info("=" * 60)
        logger.info("FALLBACK: Web Search via DuckDuckGo")
        logger.info("=" * 60)

        web_results = web_search_and_rank(query, query_embedding)
        elapsed = time.time() - start_time

        if not web_results:
            return self._error_result("No results found from any source.", start_time)

        formatted = self._format_web_results(web_results[:3])

        result = {
            "source": "web",
            "query": query,
            "results": formatted,
            "execution_time": elapsed,
        }

        # Cache web results too
        self.cache_agent.store(query, query_embedding, formatted)

        return result

    def _format_youtube_results(self, results: list[dict]) -> list[dict]:
        """Format YouTube results for output."""
        formatted = []
        for r in results:
            formatted.append({
                "rank": r.get("rank", 0),
                "title": r.get("title", "Unknown"),
                "channel": r.get("channel_name", "Unknown"),
                "timestamp_link": r.get("timestamp_url", r.get("url", "")),
                "timestamp": r.get("timestamp_str", "00:00:00"),
                "snippet": r.get("chunk_text", "")[:200],
                "similarity_score": round(r.get("similarity", 0), 4),
                "final_score": round(r.get("final_score", 0), 4),
                "view_count": r.get("view_count", 0),
            })
        return formatted

    def _format_web_results(self, results: list[dict]) -> list[dict]:
        """Format web results for output."""
        formatted = []
        for i, r in enumerate(results):
            formatted.append({
                "rank": i + 1,
                "title": r.get("title", "Unknown"),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", "")[:300],
                "similarity_score": round(r.get("similarity", 0), 4),
                "is_trusted": r.get("is_trusted", False),
            })
        return formatted

    def _evaluation_scores(self, results: list[dict]) -> dict:
        """
        Evaluation: print similarity + final_score for transparency.
        """
        if not results:
            return {}
        sim_scores = [r.get("similarity", 0) for r in results]
        final_scores = [r.get("final_score", 0) for r in results]
        return {
            "mean_similarity": round(sum(sim_scores) / len(sim_scores), 4),
            "max_similarity": round(max(sim_scores), 4),
            "min_similarity": round(min(sim_scores), 4),
            "mean_final_score": round(sum(final_scores) / len(final_scores), 4),
            "max_final_score": round(max(final_scores), 4),
        }

    def _error_result(self, message: str, start_time: float) -> dict:
        """Return an error result."""
        elapsed = time.time() - start_time
        return {
            "source": "error",
            "query": "",
            "results": [],
            "message": message,
            "execution_time": elapsed,
        }
