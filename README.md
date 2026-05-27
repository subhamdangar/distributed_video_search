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
- [OpenRouter API Key Setup](#openrouter-api-key-setup)
- [Usage](#usage)
- [Ranking System](#ranking-system)
- [Distributed Setup](#distributed-setup)
- [Semantic Cache](#semantic-cache)

---

## Overview

This system answers educational queries by searching curated YouTube channels — no YouTube Data API key required. It embeds queries using a multilingual sentence transformer, searches relevant channels in parallel across a Dask cluster, and returns top timestamped video results with direct jump links.

### Key Features

- Multilingual query support (Hindi, English, Hinglish)
- Distributed execution across multiple machines using Dask
- 3-tier retrieval pipeline (transcript → chapter → title fallback)
- Hybrid ranking system using semantic similarity + metadata + keyword overlap
- SQLite-based semantic cache
- DuckDuckGo web fallback when YouTube results are weak
- LLM-based query routing using OpenRouter

---

## Architecture

```text
User Query
    │
    ▼
┌─────────────┐
│ Query Agent │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Cache Agent │
└──────┬──────┘
       │ MISS
       ▼
┌──────────────┐
│ Router Agent │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────┐
│       Dask Distributed System      │
│                                    │
│ Worker 1   Worker 2   Worker 3     │
│ Channel A  Channel B  Channel C    │
└────────────────────────────────────┘
       │
       ▼
┌────────────────┐
│ Ranking Agent  │
└───────┬────────┘
        │
        ▼
 YouTube Results / Web Fallback
```

---

## Pipeline

| Stage | Description |
|------|-------------|
| 1 | Query cleaning and multilingual embedding |
| 2 | Semantic cache lookup |
| 3 | LLM-based subject routing |
| 4 | Parallel channel processing via Dask |
| 5 | Hybrid ranking |
| 6 | Relevance check |
| 7 | Cache storage + final response |

---

## Agent Reference

### QueryAgent

- Cleans raw user query
- Generates multilingual embeddings using:
  `paraphrase-multilingual-MiniLM-L12-v2`

---

### CacheAgent

- SQLite-based semantic cache
- Uses cosine similarity over embeddings
- Prevents redundant YouTube/web requests

---

### RouterAgent

Uses OpenRouter LLM (`Llama 3 8B`) to classify queries into:

- `mathematics`
- `computer_science`

---

### YouTubeAgent

Responsible for:

- Fetching videos using `yt-dlp`
- Metadata filtering
- Chapter/title matching
- Timestamp extraction

---

### RankingAgent

Uses hybrid scoring:

```text
final_score = 0.50 × chunk_similarity
            + 0.30 × title_similarity
            + 0.20 × keyword_score
```

---

### WebAgent

DuckDuckGo fallback search with embedding-based ranking and trusted-domain boosting.

---

## Project Structure

```text
.
├── main.py
│
├── agents/
│   ├── orchestrator.py
│   ├── query_agent.py
│   ├── cache_agent.py
│   ├── router_agent.py
│   ├── youtube_agent.py
│   ├── ranking_agent.py
│   └── web_agent.py
│
├── config/
│   └── channels.py
│
├── utils/
│   ├── cleaning.py
│   ├── embeddings.py
│   └── similarity.py
│
├── db/
│   └── cache.db
│
├── requirements.txt
└── README.md
```

---

## Supported Channels

| Subject | Channels |
|---------|----------|
| **Mathematics** | 3Blue1Brown, MIT OpenCourseWare, Khan Academy, Vedantu JEE, Physics Wallah |
| **Computer Science** | freeCodeCamp, CodeWithHarry, NPTEL |

To add channels, edit `config/channels.py`:

```python
COMPUTER_SCIENCE_CHANNELS.append({
    "name": "My Channel",
    "channel_id": "UCxxxxxxxxxxxxxxxxxx",
    "language": "hi"
})
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/distributed-video-search.git

cd distributed-video-search
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Create Database Directory

```bash
mkdir -p db
touch db/cache.db
```

## OpenRouter API Key Setup
Get a free API key from:

https://openrouter.ai/settings/keys

---
The `RouterAgent` uses the OpenRouter API for LLM-based query classification.

---

### Steps — Create a `.env` File

Create a `.env` file in the project root and add this line in this file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Example structure:

```text
.
├── .env
├── main.py
├── agents/
├── config/
├── utils/
└── requirements.txt
```




---

## Configuration

All system configuration is stored in:

```text
config/channels.py
```

### Important Constants

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



## Usage

### Single Query

```bash
python main.py "binary search tree kaise banaye"
```

```bash
python main.py "integration by parts explained"
```

---

### Interactive Mode

```bash
python main.py
```

---

### JSON Output

```bash
python main.py "quicksort algorithm" --json
```

---

### Clear Cache

```bash
python main.py --clear-cache
```

---

## Ranking System

The ranking system uses a 3-signal hybrid score.

| Signal | Weight |
|--------|--------|
| `chunk_similarity` | 50% |
| `title_similarity` | 30% |
| `keyword_score` | 20% |

View count acts only as a tiebreaker.

---

## Distributed Setup

### Start Scheduler

```bash
PYTHONPATH=$(pwd) dask scheduler
```

---

### Start Workers

```bash
PYTHONPATH=$(pwd) dask worker tcp://<scheduler_ip>:8786
```

---

### Configure Scheduler Address

Inside `config/channels.py`:

```python
DASK_SCHEDULER_ADDRESS = "tcp://<scheduler_ip>:8786"
```

---

### Run Query

```bash
python main.py "your query here"
```

---

## Semantic Cache

SQLite-based semantic cache stored at:

```text
db/cache.db
```

Schema:

```sql
CREATE TABLE semantic_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    embedding BLOB NOT NULL,
    results TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Clear cache:

```bash
python main.py --clear-cache
```