# #!/usr/bin/env python3
# """
# Distributed Multilingual Educational Search System
# ═══════════════════════════════════════════════════

# CLI entry point.

# Usage:
#   python main.py "What is the derivative of sin x?"
#   python main.py "binary search tree kaise banaye"
#   python main.py "Newton's laws of motion explained"
#   python main.py --clear-cache
#   python main.py --help

# The system will:
#   1. Understand and embed your query (Hindi/English/Hinglish)
#   2. Check semantic cache for previous results
#   3. Route to relevant subject channels
#   4. Fetch YouTube videos, transcripts, and rank them
#   5. Fall back to web search if no relevant videos found
#   6. Return top 3 timestamped results
# """

# import sys
# import os
# import argparse
# import logging
# import json

# # Ensure project root is on the path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from rich.console import Console
# from rich.panel import Panel
# from rich.table import Table
# from rich.text import Text
# from rich import box

# console = Console()


# def setup_logging(verbose: bool = False):
#     """Configure logging with Rich-compatible formatting."""
#     level = logging.DEBUG if verbose else logging.INFO
#     logging.basicConfig(
#         level=level,
#         format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
#         datefmt="%H:%M:%S",
#     )
#     # Suppress noisy third-party loggers
#     logging.getLogger("urllib3").setLevel(logging.WARNING)
#     logging.getLogger("yt_dlp").setLevel(logging.WARNING)
#     logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
#     logging.getLogger("httpx").setLevel(logging.WARNING)
#     logging.getLogger("httpcore").setLevel(logging.WARNING)
#     logging.getLogger("transformers").setLevel(logging.WARNING)
#     logging.getLogger("dask").setLevel(logging.WARNING)
#     logging.getLogger("distributed").setLevel(logging.WARNING)


# def display_youtube_results(result: dict):
#     """Display YouTube results in a rich formatted table."""
#     console.print()
#     console.print(Panel(
#         f"[bold green]🎬 YouTube Results[/bold green]\n"
#         f"[dim]Query: {result['query']}[/dim]\n"
#         f"[dim]Channels searched: {result.get('channels_searched', '?')} │ "
#         f"Chunks processed: {result.get('total_chunks_processed', '?')} │ "
#         f"Time: {result['execution_time']:.2f}s[/dim]",
#         title="[bold]Search Results[/bold]",
#         border_style="green",
#     ))

#     table = Table(
#         box=box.ROUNDED,
#         show_header=True,
#         header_style="bold cyan",
#         expand=True,
#     )
#     table.add_column("#", style="bold", width=3)
#     table.add_column("Title", style="bold white", ratio=4)
#     table.add_column("Channel", style="yellow", ratio=2)
#     table.add_column("Timestamp", style="green", max_width=12)
#     table.add_column("Views", style="magenta", justify="right", max_width=12)
#     table.add_column("Score", style="cyan", justify="right", max_width=8)

#     for r in result["results"]:
#         views = r.get("view_count", 0)
#         views_str = f"{views:,}" if views else "N/A"
#         table.add_row(
#             str(r.get("rank", "")),
#             r.get("title", "Unknown"),
#             r.get("channel", "Unknown"),
#             r.get("timestamp", "00:00:00"),
#             views_str,
#             str(r.get("similarity_score", 0)),
#         )

#     console.print(table)

#     # Show details with FULL URLs (not truncated!)
#     for r in result["results"]:
#         link = r.get("timestamp_link", "")
#         snippet = r.get("snippet", "No snippet available")
#         console.print(Panel(
#             f"[bold blue]🔗 {link}[/bold blue]\n\n"
#             f"[dim]{snippet}[/dim]",
#             title=f"[bold]#{r.get('rank', '')} — {r.get('title', '')}[/bold]",
#             border_style="dim",
#             padding=(0, 2),
#         ))

#     # Show evaluation scores
#     if "scores" in result and result["scores"]:
#         scores = result["scores"]
#         console.print(f"\n[dim]📊 Evaluation — "
#                        f"Mean: {scores.get('mean_similarity', 0):.4f} │ "
#                        f"Max: {scores.get('max_similarity', 0):.4f} │ "
#                        f"Min: {scores.get('min_similarity', 0):.4f}[/dim]")


# def display_web_results(result: dict):
#     """Display web search results in a rich formatted panel."""
#     console.print()
#     console.print(Panel(
#         f"[bold yellow]🌐 Web Search Results (Fallback)[/bold yellow]\n"
#         f"[dim]Query: {result['query']}[/dim]\n"
#         f"[dim]Time: {result['execution_time']:.2f}s[/dim]",
#         title="[bold]Search Results[/bold]",
#         border_style="yellow",
#     ))

#     table = Table(
#         box=box.ROUNDED,
#         show_header=True,
#         header_style="bold cyan",
#         expand=True,
#     )
#     table.add_column("#", style="bold", width=3)
#     table.add_column("Title", style="bold white", ratio=4)
#     table.add_column("Score", style="cyan", justify="right", max_width=8)
#     table.add_column("Trusted", style="green", justify="center", max_width=8)

#     for r in result["results"]:
#         trusted = "✅" if r.get("is_trusted", False) else ""
#         table.add_row(
#             str(r.get("rank", "")),
#             r.get("title", "Unknown"),
#             str(r.get("similarity_score", 0)),
#             trusted,
#         )

#     console.print(table)

#     # Show details with FULL URLs
#     for r in result["results"]:
#         url = r.get("url", "")
#         snippet = r.get("snippet", "No snippet available")
#         console.print(Panel(
#             f"[bold blue]🔗 {url}[/bold blue]\n\n"
#             f"[dim]{snippet}[/dim]",
#             title=f"[bold]#{r.get('rank', '')} — {r.get('title', '')}[/bold]",
#             border_style="dim",
#             padding=(0, 2),
#         ))


# def display_cache_results(result: dict):
#     """Display cached results."""
#     console.print()
#     console.print(Panel(
#         f"[bold blue]⚡ Cached Results[/bold blue]\n"
#         f"[dim]Query: {result['query']}[/dim]\n"
#         f"[dim]Matched cached query: {result.get('cached_query', '?')}[/dim]\n"
#         f"[dim]Cache similarity: {result.get('cache_similarity', 0):.4f} │ "
#         f"Time: {result['execution_time']:.2f}s[/dim]",
#         title="[bold]Cached Search Results[/bold]",
#         border_style="blue",
#     ))

#     # Detect result type and display accordingly
#     results = result.get("results", [])
#     if results and "timestamp_link" in results[0]:
#         # YouTube-style cached results
#         for r in results:
#             console.print(Panel(
#                 f"[bold]{r.get('title', 'Unknown')}[/bold]\n"
#                 f"Channel: {r.get('channel', 'Unknown')} │ "
#                 f"Timestamp: {r.get('timestamp', '00:00:00')}\n"
#                 f"🔗 {r.get('timestamp_link', '')}\n\n"
#                 f"[dim]{r.get('snippet', '')}[/dim]",
#                 border_style="blue",
#             ))
#     else:
#         for r in results:
#             console.print(Panel(
#                 f"[bold]{r.get('title', 'Unknown')}[/bold]\n"
#                 f"🔗 {r.get('url', '')}\n\n"
#                 f"[dim]{r.get('snippet', '')}[/dim]",
#                 border_style="blue",
#             ))


# def display_error(result: dict):
#     """Display error message."""
#     console.print()
#     console.print(Panel(
#         f"[bold red]❌ No Results Found[/bold red]\n"
#         f"[dim]{result.get('message', 'Unknown error')}[/dim]\n"
#         f"[dim]Time: {result['execution_time']:.2f}s[/dim]",
#         title="[bold]Error[/bold]",
#         border_style="red",
#     ))


# def main():
#     parser = argparse.ArgumentParser(
#         description="Distributed Multilingual Educational Search System",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python main.py "calculus integration by parts"
#   python main.py "binary search tree kaise banate hain"
#   python main.py "quantum mechanics explained" --verbose
#   python main.py "organic chemistry reaction mechanisms" --json
#   python main.py --clear-cache
#         """,
#     )
#     parser.add_argument(
#         "query",
#         nargs="?",
#         help="Your search query in Hindi, English, or Hinglish",
#     )
#     parser.add_argument(
#         "--verbose", "-v",
#         action="store_true",
#         help="Enable verbose/debug logging",
#     )
#     parser.add_argument(
#         "--json",
#         action="store_true",
#         help="Output results as JSON instead of formatted text",
#     )
#     parser.add_argument(
#         "--clear-cache",
#         action="store_true",
#         help="Clear the semantic cache and exit",
#     )

#     args = parser.parse_args()
#     setup_logging(args.verbose)

#     # ── Handle cache clearing ─────────────────────────────────
#     if args.clear_cache:
#         from agents.cache_agent import CacheAgent
#         cache = CacheAgent()
#         cache.clear()
#         console.print("[green]✅ Cache cleared successfully.[/green]")
#         return

#     # ── Validate query ────────────────────────────────────────
#     if not args.query:
#         parser.print_help()
#         console.print("\n[red]Error: Please provide a search query.[/red]")
#         sys.exit(1)

#     # ── Display banner ────────────────────────────────────────
#     if not args.json:
#         console.print(Panel(
#             "[bold cyan]Distributed Multilingual Educational Search System[/bold cyan]\n"
#             "[dim]Powered by Dask · LangChain · Sentence Transformers[/dim]",
#             border_style="cyan",
#             padding=(1, 4),
#         ))
#         console.print(f"\n🔍 Searching for: [bold]{args.query}[/bold]\n")

#     # ── Run the pipeline ──────────────────────────────────────
#     from agents.orchestrator import Orchestrator

#     orchestrator = Orchestrator()
#     result = orchestrator.search(args.query)

#     # ── Display results ───────────────────────────────────────
#     if args.json:
#         print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
#     else:
#         source = result.get("source", "error")
#         if source == "youtube":
#             display_youtube_results(result)
#         elif source == "web":
#             display_web_results(result)
#         elif source == "cache":
#             display_cache_results(result)
#         else:
#             display_error(result)

#         console.print(f"\n[dim]Total execution time: {result['execution_time']:.2f}s[/dim]\n")


# if __name__ == "__main__":
#     main()























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
            print(f"Score: {r.get('similarity_score')}")
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
    orchestrator = Orchestrator()

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

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_results(result)


if __name__ == "__main__":
    main()