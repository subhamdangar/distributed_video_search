# Distributed Multilingual Educational Video Search System

A distributed search engine for educational YouTube content, supporting **Hindi**, **English**, and **Hinglish** queries. Built on **Dask** for multi-machine parallel execution, it retrieves timestamped YouTube results ranked by a 3-signal hybrid scoring model, with automatic fallback to web search via DuckDuckGo.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Agent Reference](#agent-reference)
- [Project Structure](#project-structure)
- [Supported Channels](#supported-channels)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Ranking System](#ranking-system)
- [Distributed Setup](#distributed-setup)
- [Semantic Cache](#semantic-cache)

---

## Overview

This system answers educational queries by searching curated YouTube channels — no YouTube Data API key required. It embeds queries using a multilingual sentence transformer, searches relevant channels in parallel across a Dask cluster, and returns top timestamped video results with direct jump links.

**Key capabilities:**

- Multilingual queries — Hindi (Devanagari), English, Hinglish (code-mixed)
- Distributed execution across multiple machines via Dask
- 3-tier video matching: transcript → chapter → title fallback
- 3-signal hybrid ranking: semantic similarity + title match + keyword overlap
- Semantic cache (SQLite + cosine similarity) to skip redundant fetches
- Web search fallback via DuckDuckGo when YouTube results are insufficient
- LLM-based query routing via OpenRouter (free tier)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ Query Agent │  ── clean + embed (multilingual MiniLM-L12-v2)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Cache Agent │  ── semantic similarity search (SQLite)
└──────┬──────┘
       │  MISS
       ▼
┌──────────────┐
│ Router Agent │  ── LLM classification → subject (mathematics / computer_science / physics / chemistry)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│              Dask Distributed Scheduler           │
│                                                  │
│   Worker 1        Worker 2        Worker 3  ...  │
│  [Channel A]     [Channel B]     [Channel C]     │
│                                                  │
│  Each worker runs process_channel():             │
│    Stage 1: Metadata filter (embed all titles)   │
│    Stage 2: Deep match (chapters → title)        │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Ranking Agent  │  ── 3-signal hybrid scoring + deduplication
              └───────┬────────┘
                      │
          ┌───────────┴──────────┐
          │ Relevant?            │ No → Web Agent (DuckDuckGo fallback)
          │ Yes                  │
          ▼                      ▼
   YouTube Results          Web Results
   (timestamped links)      (ranked URLs)
          │
          ▼
   Cache Store → Return to User
```

---

## Pipeline

The `Orchestrator` runs a 7-stage pipeline on every query:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Query Understanding | Clean and embed the raw query using multilingual MiniLM |
| 2 | Semantic Cache Check | Cosine similarity search against cached queries (threshold: 0.78) |
| 3 | Topic Routing | LLM classifies query into a subject via OpenRouter |
| 4 | Dask Parallel Execution | Each channel processed concurrently across workers |
| 5 | Ranking | 3-signal hybrid scoring with deduplication |
| 6 | Relevance Check | If best score < 0.45, fall back to web search |
| 7 | Cache + Return | Store result in cache, return formatted output |

---

## Agent Reference

### `QueryAgent`
Cleans raw user input and generates a 384-dimensional multilingual embedding using `paraphrase-multilingual-MiniLM-L12-v2`. Supports Hindi, English, and Hinglish.

### `CacheAgent`
SQLite-backed semantic cache. On each query, computes cosine similarity against all cached query embeddings. Returns cached results if similarity ≥ threshold (default: `0.78`), skipping YouTube/web entirely.

### `RouterAgent`
Sends the cleaned query to an LLM (Llama 3 8B via OpenRouter) with a strict classification prompt. Returns the subject (`mathematics`, `computer_science`, `physics`, `chemistry`) or triggers web fallback if the classification is ambiguous.

### `YouTubeAgent` (`process_channel`)
Runs per-channel on each Dask worker. Two-stage retrieval:

- **Stage 1 — Metadata Filter:** Fetches all channel videos via `yt-dlp`. Embeds titles + descriptions and scores against the query. Keeps top 20 with score ≥ 0.25.
- **Stage 2 — Deep Match (3-tier):**
  - **Tier 1 — Transcript** *(currently disabled, available in code)*: Chunk and embed full transcript for precise timestamp matching.
  - **Tier 2 — Chapters:** Parse chapter markers or description timestamps. Embed chapter titles and match against query.
  - **Tier 3 — Title:** Match query against title + description snippet. Timestamp defaults to `00:00:00`.

### `RankingAgent`
Deduplicates results (one per video), computes a **hybrid final score**, and returns top K:

```
final_score = 0.50 × chunk_similarity
            + 0.30 × title_similarity
            + 0.20 × keyword_score
```

View count is used only as a tiebreaker — it does not override relevance.

### `WebAgent`
DuckDuckGo-based fallback. Searches, loads page content via LangChain `WebBaseLoader` (with BeautifulSoup fallback), chunks, embeds, and ranks. Trusted educational sites (Wikipedia, GeeksForGeeks, Khan Academy, etc.) receive a 1.15× similarity boost. Supports Hinglish queries via automatic English keyword extraction and retry.

---

## Project Structure

```
.
├── main.py                        # CLI entry point (interactive + single-query mode)
│
├── agents/
│   ├── orchestrator.py            # Master pipeline controller
│   ├── query_agent.py             # Query cleaning + embedding
│   ├── cache_agent.py             # Semantic cache (SQLite)
│   ├── router_agent.py            # LLM-based topic routing (OpenRouter)
│   ├── youtube_agent.py           # YouTube retrieval (yt-dlp, 3-tier matching)
│   ├── ranking_agent.py           # 3-signal hybrid ranking
│   └── web_agent.py               # DuckDuckGo web fallback
│
├── config/
│   └── channels.py                # Channel lists, system constants, Dask config
│
├── utils/
│   ├── cleaning.py                # Text cleaning + chunking utilities
│   ├── embeddings.py              # Sentence transformer wrapper (singleton)
│   └── similarity.py             # Cosine similarity utilities
│
├── db/
│   └── cache.db                   # SQLite semantic cache (auto-created)
│
└── requirements.txt
```

---

## Supported Channels

| Subject | Channels |
|---------|----------|
| **Mathematics** | 3Blue1Brown, MIT OpenCourseWare, Khan Academy, Vedantu JEE, Physics Wallah |
| **Computer Science** | freeCodeCamp, CodeWithHarry, Apna College |
| **Physics** | Physics Wallah, MIT OpenCourseWare, Vedantu JEE |
| **Chemistry** | Khan Academy, The Organic Chemistry Tutor, Vedantu JEE, Physics Wallah |

To add channels, edit `config/channels.py`:

```python
COMPUTER_SCIENCE_CHANNELS.append({
    "name": "My Channel",
    "channel_id": "UCxxxxxxxxxxxxxxxxxx",
    "language": "hi",   # "en" | "hi" | "mixed"
})
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/distributed-video-search.git
cd distributed-video-search

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create the cache database directory
mkdir -p db
touch db/cache.db
```

---

## Configuration

All configuration lives in `config/channels.py`.

| Constant | Default | Description |
|----------|---------|-------------|
| `DASK_SCHEDULER_ADDRESS` | `None` | Set to `tcp://<IP>:8786` for distributed mode |
| `CACHE_SIMILARITY_THRESHOLD` | `0.78` | Cosine similarity threshold for cache hit |
| `TOP_SIMILARITY_RESULTS` | `10` | Candidates retained after scoring |
| `TOP_FINAL_RESULTS` | `3` | Final results returned to user |
| `TRANSCRIPT_CHUNK_SIZE` | `500` | Characters per transcript chunk |
| `TRANSCRIPT_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer model |
| `WEB_SEARCH_MAX_URLS` | `5` | DuckDuckGo results to process in fallback |
| `TRUSTED_SITE_BOOST` | `1.15` | Similarity multiplier for trusted domains |

**OpenRouter API key** (free tier, required for routing):

```python
# agents/router_agent.py
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxxxxxx"
```

Get a free key at [openrouter.ai](https://openrouter.ai).

---

## Usage

**Single query:**
```bash
python main.py "binary search tree kaise banaye"
python main.py "integration by parts explained"
python main.py "Newton's laws of motion"
```

**Interactive mode:**
```bash
python main.py
# → Enter query: binary search kya hota hai
```

**JSON output:**
```bash
python main.py "quicksort algorithm" --json
```

**Clear semantic cache:**
```bash
python main.py --clear-cache
```

**Example output:**
```
==============================
RESULT
==============================

Query: binary search tree kaise banaye
Source: youtube
Time: 8.43s

[1] Data Structures: Binary Search Tree
Channel: Apna College
Timestamp: 00:03:24
Link: https://www.youtube.com/watch?v=xxxxx&t=204s
Score: 0.7821  |  Final: 0.6534
Views: 1,204,839
--------------------------------------------------
```

---

## Ranking System

The `RankingAgent` uses a **3-signal hybrid score** to balance deep semantic match with intent awareness and exact keyword coverage:

| Signal | Weight | Source |
|--------|--------|--------|
| `chunk_similarity` | 50% | Cosine similarity of matched chapter/chunk embedding vs query |
| `title_similarity` | 30% | Cosine similarity of video title+description vs query |
| `keyword_score` | 20% | Fraction of query words found in video text |

View count acts as a tiebreaker only — a viral but irrelevant video cannot outrank a deeply relevant one.

**Fallback threshold:** If the best `final_score` is below `0.45`, the system discards YouTube results and invokes the web search fallback.

---

## Distributed Setup

The system supports multi-machine execution via **Dask Distributed**. Each channel is submitted as an independent task to the scheduler and executed on whichever worker is available.

**Machine 1 — Scheduler + Client:**
```bash
PYTHONPATH=$(pwd) dask scheduler
# Scheduler at: tcp://<IP>:8786
# Dashboard at: http://<IP>:8787
```

**Machines 2, 3, 4... — Workers:**
```bash
PYTHONPATH=$(pwd) dask worker tcp://<scheduler_ip>:8786
```

**Set the scheduler address in `config/channels.py`:**
```python
DASK_SCHEDULER_ADDRESS = "tcp://<scheduler_ip>:8786"
```

**Run on the Scheduler machine:**
```bash
python main.py "your query here"
```

> **Always use `PYTHONPATH=$(pwd)` inline** — not via `export`. This guarantees the worker process inherits the correct module path.

If the scheduler is unreachable, the system automatically falls back to local threaded execution using `dask.delayed`.

---

## Semantic Cache

The `CacheAgent` maintains a persistent SQLite database at `db/cache.db`.

**Schema:**

```sql
CREATE TABLE semantic_cache (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query      TEXT NOT NULL,
    embedding  BLOB NOT NULL,      -- numpy float32 array
    results    TEXT NOT NULL,      -- JSON-serialized result
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

On every new query, the system computes cosine similarity between the query embedding and all stored embeddings. If the best match exceeds the threshold (default `0.78`), cached results are returned immediately — no YouTube or web requests are made.

**Clear the cache:**
```bash
python main.py --clear-cache
```
