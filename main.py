#!/usr/bin/env python3

import sys
import os
import argparse
import logging
import json

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(verbose: bool = True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("dask").setLevel(logging.WARNING)


def print_results(result: dict):
    source = result.get("source", "error")

    print("\n==============================")
    print("RESULT")
    print("==============================\n")

    print(f"Query: {result.get('query', '')}")
    print(f"Source: {source}")
    print(f"Time: {result.get('execution_time', 0):.2f}s\n")

    if source == "youtube":
        for r in result.get("results", []):
            print(f"[{r.get('rank')}] {r.get('title')}")
            print(f"Channel: {r.get('channel')}")
            print(f"Timestamp: {r.get('timestamp')}")
            print(f"Link: {r.get('timestamp_link')}")
            print(f"Score: {r.get('similarity_score')}  |  Final: {r.get('final_score')}")
            print(f"Views: {r.get('view_count')}")
            print(f"Snippet: {r.get('snippet')}")
            print("-" * 50)

    elif source == "web":
        for r in result.get("results", []):
            print(f"[{r.get('rank')}] {r.get('title')}")
            print(f"URL: {r.get('url')}")
            print(f"Score: {r.get('similarity_score')}")
            print(f"Snippet: {r.get('snippet')}")
            print("-" * 50)

    elif source == "cache":
        print(f"Cached query: {result.get('cached_query')}")
        print(f"Similarity: {result.get('cache_similarity')}\n")

        for r in result.get("results", []):
            print(f"[{r.get('rank')}] {r.get('title')}")
            print(f"Channel: {r.get('channel')}")
            print(f"Link: {r.get('timestamp_link')}")
            print(f"{r.get('snippet')}")
            print("-" * 50)

    else:
        print("Error:", result.get("message", "Unknown error"))


def main():
    parser = argparse.ArgumentParser(description="Distributed Search System")

    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # ── Clear cache ───────────────────────────────
    if args.clear_cache:
        from agents.cache_agent import CacheAgent
        cache = CacheAgent()
        cache.clear()
        print("Cache cleared.")
        return

    # ── Load orchestrator ─────────────────────────
    from agents.orchestrator import Orchestrator
    orchestrator = Orchestrator()             #-----------------------------------------> Call the orchestrator.py

    # ── If query passed via CLI ───────────────────
    if args.query:
        result = orchestrator.search(args.query)

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_results(result)
        return

    # ── Interactive mode ──────────────────────────
    print("=== Distributed Educational Search System ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Enter query: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not query:
            print("Empty query. Try again.")
            continue

        result = orchestrator.search(query)

        # if args.json:
        #     print(json.dumps(result, indent=2, ensure_ascii=False))
        # else:
        #     print_results(result)
        
        print_results(result)


if __name__ == "__main__":
    main()