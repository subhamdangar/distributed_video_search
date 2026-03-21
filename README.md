# Distributed Multilingual Educational Search System

A production-quality, distributed educational search system that retrieves and ranks relevant YouTube video timestamps and web content for natural-language queries in **Hindi**, **English**, and **Hinglish** (code-mixed).

> **No paid APIs required.** Uses DuckDuckGo search, YouTube transcripts, and open-source multilingual embeddings.

---

## 🏗️ Architecture Overview

```
User Query (Hindi / English / Hinglish)
    │
    ▼
┌──────────────────────────┐
│  Query Understanding     │  ← Clean + Embed (multilingual MiniLM)
│  Agent                   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Cache Agent             │  ← Semantic similarity search (SQLite)
│  (Semantic Cache)        │
└──────────┬───────────────┘
           │  MISS
           ▼
┌──────────────────────────┐
│  Router Agent            │  ← Keyword + Embedding classification
│  (Topic Classifier)      │     → math / cs / physics / chemistry
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  Dask Distributed Execution                          │
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │ Node 1  │  │ Node 2  │  │ Node 3  │  │ Node 4 │ │
│  │  Math   │  │   CS    │  │ Physics │  │  Chem  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └───┬────┘ │
│       │            │            │            │      │
│       ▼            ▼            ▼            ▼      │
│  ┌──────────────────────────────────────────────┐   │
│  │  YouTube Retrieval Agent (per channel)       │   │
│  │  fetch → transcript → chunk → embed → score │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────┐
│  Ranking Agent           │  ← Top 10 by similarity → Top 3 by views
└──────────┬───────────────┘
           │
           ▼  (If no relevant results)
┌──────────────────────────┐
│  Web Search Agent        │  ← DuckDuckGo → WebBaseLoader → Rank
│  (Fallback)              │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Cache Store + Output    │
└──────────────────────────┘
```

---

## 📂 Project Structure

```
project/
├── main.py                    # CLI entry point
├── config/
│   ├── __init__.py
│   └── channels.py            # Channel IDs, subjects, and constants
├── agents/
│   ├── __init__.py
│   ├── query_agent.py         # Query cleaning + embedding
│   ├── cache_agent.py         # SQLite semantic cache
│   ├── router_agent.py        # Topic classification
│   ├── youtube_agent.py       # Video fetching + transcript processing
│   ├── web_agent.py           # DuckDuckGo fallback
│   ├── ranking_agent.py       # Two-step ranking (similarity + views)
│   └── orchestrator.py        # Pipeline controller (Dask)
├── utils/
│   ├── __init__.py
│   ├── embeddings.py          # Multilingual MiniLM embeddings
│   ├── similarity.py          # Cosine similarity + top-k
│   └── cleaning.py            # Text cleaning + chunking
├── db/
│   └── cache.db               # SQLite cache (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### 1. Clone / Navigate to the Project

```bash
cd /path/to/Distributed_Computing_Project
```

### 2. Create Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate    # macOS/Linux
# myenv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the multilingual MiniLM model (~470 MB). This is cached for subsequent runs.

### 4. Run the System

```bash
# Basic usage
python main.py "What is integration by parts?"

# Hindi query
python main.py "न्यूटन के गति के नियम समझाइए"

# Hinglish (code-mixed) query
python main.py "binary search tree kaise banaye"

# Verbose mode (shows full pipeline logs)
python main.py "quantum mechanics explained" --verbose

# JSON output
python main.py "organic chemistry reactions" --json

# Clear cache
python main.py --clear-cache
```

---

## 🧩 Module Descriptions

### 1. Query Understanding Agent (`agents/query_agent.py`)
- **Input:** Raw user query (any language)
- **Output:** Cleaned query string + 384-dimensional embedding
- Uses `paraphrase-multilingual-MiniLM-L12-v2` from sentence-transformers
- Handles Hindi (Devanagari), English, and code-mixed text

### 2. Cache Agent (`agents/cache_agent.py`)
- SQLite-backed semantic cache
- Stores query embeddings as BLOBs alongside JSON results
- On new queries, computes cosine similarity against all cached embeddings
- Returns cached results if similarity ≥ 0.88 (configurable)
- Dramatically reduces API calls for repeated/similar queries

### 3. Router Agent (`agents/router_agent.py`)
- Classifies queries into subjects: `mathematics`, `computer_science`, `physics`, `chemistry`
- **Hybrid approach:**
  - First tries keyword matching (English + Hindi keywords)
  - Falls back to embedding similarity against rich subject descriptions
  - If confidence is low, routes to ALL subjects (ranking agent filters)

### 4. YouTube Retrieval Agent (`agents/youtube_agent.py`)
- Fetches videos from configured channels via `yt_dlp`
- Loads transcripts using LangChain `YoutubeLoader` (fallback: `youtube_transcript_api`)
- Chunks transcripts (~30 sec segments) with character-offset tracking
- Embeds each chunk and computes similarity to the query
- Estimates timestamps from character positions relative to video duration

### 5. Ranking Agent (`agents/ranking_agent.py`)
- **Step 1:** Select top 10 results by cosine similarity
- **Step 2:** From top 10, select top 3 by view count (popularity)
- Deduplicates: if multiple chunks from the same video, keeps the best chunk
- Outputs relevance threshold check for triggering web fallback

### 6. Web Search Agent (`agents/web_agent.py`)
- Activated when no relevant YouTube results are found
- Searches DuckDuckGo (free, no API key needed)
- Loads top 5 page contents via LangChain `WebBaseLoader` (fallback: BeautifulSoup)
- Chunks, embeds, and ranks web content
- **Trusted site boosting:** educational domains (Wikipedia, Khan Academy, GeeksforGeeks, etc.) get a 1.15x score multiplier

### 7. Orchestrator (`agents/orchestrator.py`)
- Controls the full 7-stage pipeline
- Uses **Dask** `delayed` + `compute` for parallel channel processing
- Threaded scheduler with configurable max workers (default: 6)
- Handles fallback logic and error recovery
- Formats results and stores in cache

---

## 📡 Distributed Nodes (Subjects)

Each subject acts as a distributed "node" with predefined YouTube channels:

| Node | Subject | Channels |
|------|---------|----------|
| 1 | Mathematics | 3Blue1Brown, MIT OCW, Khan Academy |
| 2 | Computer Science | freeCodeCamp, CodeWithHarry, NPTEL |
| 3 | Physics | Physics Wallah, MIT OCW |
| 4 | Chemistry | Unacademy, Khan Academy |

Channels are configured in `config/channels.py` with real YouTube channel IDs. To add or remove channels, simply edit the lists.

---

## ⚡ Dask Parallelism

The system parallelizes:
- **Channel processing** — each channel runs as a separate Dask delayed task
- **Video processing** — within each channel, videos are processed sequentially (to respect rate limits)
- **Embedding tasks** — batch embedding via sentence-transformers

```python
# From orchestrator.py
delayed_tasks = [delayed(process_channel)(ch, query, embedding) for ch in channels]
results = compute(*delayed_tasks, scheduler="threads", num_workers=6)
```

---

## 📊 Output Format

### YouTube Results
```
┌───┬──────────────────────────────────┬────────────┬──────────┬────────┬─────────────────────┐
│ # │ Title                            │ Channel    │ Timestamp│ Views  │ Link                │
├───┼──────────────────────────────────┼────────────┼──────────┼────────┼─────────────────────┤
│ 1 │ Integration by Parts Explained   │ 3Blue1Brown│ 00:02:15 │ 2.1M  │ youtube.com/...&t=… │
│ 2 │ Calculus: Integration Techniques │ MIT OCW    │ 00:15:30 │ 850K  │ youtube.com/...&t=… │
│ 3 │ Integration Methods Tutorial     │ Khan Acad. │ 00:04:00 │ 500K  │ youtube.com/...&t=… │
└───┴──────────────────────────────────┴────────────┴──────────┴────────┴─────────────────────┘
```

### Web Results (Fallback)
```
┌───┬──────────────────────────────────┬───────┬─────────┬─────────────────────────────────┐
│ # │ Title                            │ Score │ Trusted │ URL                             │
├───┼──────────────────────────────────┼───────┼─────────┼─────────────────────────────────┤
│ 1 │ Integration by Parts - Wikipedia │ 0.82  │   ✅    │ en.wikipedia.org/wiki/...       │
└───┴──────────────────────────────────┴───────┴─────────┴─────────────────────────────────┘
```

---

## 🔧 Configuration

All configurable parameters are in `config/channels.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_VIDEOS_PER_CHANNEL` | 8 | Max videos fetched per channel |
| `MAX_DASK_WORKERS` | 6 | Max parallel Dask threads |
| `CACHE_SIMILARITY_THRESHOLD` | 0.88 | Cosine similarity for cache hit |
| `TOP_SIMILARITY_RESULTS` | 10 | Results after similarity ranking |
| `TOP_FINAL_RESULTS` | 3 | Final results after view ranking |
| `WEB_SEARCH_MAX_URLS` | 5 | Max URLs for web fallback |
| `TRANSCRIPT_CHUNK_SIZE` | 500 | Chars per transcript chunk (~30s) |
| `TRUSTED_SITE_BOOST` | 1.15 | Score multiplier for edu sites |

---

## 📝 Important Notes

- **No paid APIs** — Everything uses free, open-source tools
- **Rate limiting** — The system uses caching and limits video fetches to be respectful
- **CPU-only** — Designed to run without GPU; sentence-transformers works well on CPU
- **First run** — Downloading the embedding model takes a few minutes (cached after)
- **Transcript availability** — Not all YouTube videos have transcripts; the system handles this gracefully

---

## 🧪 Evaluation

The system prints evaluation scores for transparency:
- **Mean similarity:** Average cosine similarity of returned results
- **Max similarity:** Best match score
- **Min similarity:** Lowest match in the returned set

---

## 📜 License

This project is for educational purposes.
