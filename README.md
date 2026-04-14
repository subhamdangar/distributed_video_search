# Distributed Multilingual Educational Video Search System

A **distributed educational search system** built with **Python** and **Dask** that retrieves and ranks relevant YouTube video timestamps for natural-language queries in **Hindi**, **English**, and **Hinglish** (code-mixed).

The system uses a multi-agent architecture where each agent handles a specific stage of the search pipeline. Channels are processed in parallel across multiple machines using **Dask Distributed**, and results are ranked using a **3-signal hybrid scoring model**.

> **No paid APIs required.** Uses DuckDuckGo search, YouTube transcripts, and open-source multilingual embeddings.

---

## Table of Contents

- [Architecture Overview](#-architecture-overview)
- [System Pipeline](#-system-pipeline)
- [Project Structure](#-project-structure)
- [Setup Instructions (Single Machine)](#-setup-instructions-single-machine)
- [Running on Multiple Machines (Lab Setup)](#-running-on-multiple-machines-lab-setup)
- [Usage](#-usage)
- [Module Descriptions](#-module-descriptions)
- [Ranking Algorithm](#-ranking-algorithm)
- [Distributed Nodes (Subjects)](#-distributed-nodes-subjects)
- [Configuration Reference](#-configuration-reference)
- [Output Format](#-output-format)
- [Troubleshooting](#-troubleshooting)
- [Important Notes](#-important-notes)
- [License](#-license)

---

## 🏗️ Architecture Overview

### High-Level System Architecture

```
+-------------------------------------------------------------------+
|                        USER QUERY                                  |
|         (English / Hindi / Hinglish)                               |
|         "what is nextjs" / "binary search tree kaise banaye"       |
+-----------------------------------+-------------------------------+
                                    |
                                    ▼
+-------------------------------------------------------------------+
|                    MAIN MACHINE (Client + Scheduler)               |
|                                                                    |
|  ┌─────────────┐   ┌──────────┐   ┌──────────────┐               |
|  │ Query Agent │──▶│ Cache    │──▶│ Router Agent │               |
|  │ Clean+Embed │   │ Agent    │   │ Classify     │               |
|  └─────────────┘   └──────────┘   └──────┬───────┘               |
|                                           │                        |
|                      ┌────────────────────┼────────────────┐       |
|                      ▼                    ▼                ▼       |
|              ┌──────────────┐   ┌──────────────┐  ┌────────────┐  |
|              │ mathematics  │   │ comp_science │  │  physics   │  |
|              │ 5 channels   │   │ 3 channels   │  │ 3 channels │  |
|              └──────┬───────┘   └──────┬───────┘  └─────┬──────┘  |
+-------------------------------------------------------------------+
                      │                  │                 │
          ┌───────────┘                  │                 └──────┐
          ▼                              ▼                        ▼
+-----------------+            +-----------------+       +-----------------+
| WORKER MACHINE 1|            | WORKER MACHINE 2|       | WORKER MACHINE 3|
|                 |            |                 |       |                 |
| process_channel |            | process_channel |       | process_channel |
| (3Blue1Brown)   |            | (freeCodeCamp)  |       | (MIT OCW)       |
|                 |            |                 |       |                 |
| ┌─────────────┐ |            | ┌─────────────┐ |       | ┌─────────────┐ |
| │Fetch ALL    │ |            | │Fetch ALL    │ |       | │Fetch ALL    │ |
| │videos       │ |            | │videos       │ |       | │videos       │ |
| │             │ |            | │             │ |       | │             │ |
| │Metadata     │ |            | │Metadata     │ |       | │Metadata     │ |
| │Filter→Top20 │ |            | │Filter→Top20 │ |       | │Filter→Top20 │ |
| │             │ |            | │             │ |       | │             │ |
| │Deep Process │ |            | │Deep Process │ |       | │Deep Process │ |
| │Transcript/  │ |            | │Transcript/  │ |       | │Transcript/  │ |
| │Chapter/Title│ |            | │Chapter/Title│ |       | │Chapter/Title│ |
| └─────────────┘ |            | └─────────────┘ |       | └─────────────┘ |
+---------+-------+            +---------+-------+       +--------+--------+
          │                              │                         │
          └──────────────┬───────────────┘                         │
                         ▼                                         │
+-------------------------------------------------------------------+
|                    MAIN MACHINE (Aggregation)                      |
|                                                                    |
|  ┌───────────────────────────────────────────────────────────┐    |
|  │              Ranking Agent (3-Signal Hybrid)              │    |
|  │                                                           │    |
|  │  final_score = 0.50 × chunk_similarity                    │    |
|  │              + 0.30 × title_similarity                    │    |
|  │              + 0.20 × keyword_score                       │    |
|  │                                                           │    |
|  │  Sort by: final_score (primary) → views (tiebreaker)      │    |
|  └───────────────┬───────────────────────────────────────────┘    |
|                  │                                                 |
|          ┌───────┴───────┐                                        |
|          ▼               ▼                                        |
|   final_score ≥ 0.45   final_score < 0.45                        |
|   ┌──────────┐         ┌───────────────┐                         |
|   │ Return   │         │ Web Fallback  │                         |
|   │ YouTube  │         │ (DuckDuckGo)  │                         |
|   │ Results  │         └───────────────┘                         |
|   └──────────┘                                                    |
+-------------------------------------------------------------------+
```

### Distributed Execution Model

```
                    ┌─────────────────────────┐
                    │     Dask Scheduler       │
                    │   (tcp://IP:8786)        │
                    │   Dashboard: :8787       │
                    └────────┬────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  Worker 1    │ │  Worker 2    │ │  Worker 3    │
    │  (Lab PC 1)  │ │  (Lab PC 2)  │ │  (Lab PC 3)  │
    │              │ │              │ │              │
    │ Runs:        │ │ Runs:        │ │ Runs:        │
    │ process_     │ │ process_     │ │ process_     │
    │ channel()    │ │ channel()    │ │ channel()    │
    └──────────────┘ └──────────────┘ └──────────────┘

    Client (main.py) runs on the SAME machine as the Scheduler.
    Client submits tasks → Scheduler distributes → Workers execute → Results gathered.
```

---

## 🔄 System Pipeline

The search pipeline has **7 stages**, executed in order:

### Stage 1: Query Understanding
```
Raw Query → Clean (remove noise) → Embed (384-dim multilingual vector)
```
- Uses `paraphrase-multilingual-MiniLM-L12-v2` model
- Supports Hindi (Devanagari), English, and code-mixed text

### Stage 2: Semantic Cache Check
```
Query Embedding → Compare against cached embeddings → If similarity ≥ 0.89 → CACHE HIT
```
- SQLite-backed semantic cache
- Avoids re-processing identical/similar queries

### Stage 3: Topic Routing
```
Query → Keyword Match + Embedding Similarity → Route to subjects (math/cs/physics/chem)
```
- Hybrid: keyword-based (English + Hindi keywords) + embedding similarity
- If confidence is low → routes to ALL subjects

### Stage 4: Dask Distributed Execution
```
For each channel (in parallel across worker machines):
  1. Fetch ALL videos from the channel (full metadata, no limit)
  2. Score ALL videos using title+description embeddings vs query
  3. Compute keyword overlap score
  4. Filter: drop videos with quick_score < 0.25
  5. Select top-20 candidates
  6. Deep process each candidate:
     - Tier 1: Try transcript → chunk → embed → match (precise timestamps)
     - Tier 2: Try chapter markers → embed → match (accurate timestamps)
     - Tier 3: Title-only fallback (timestamp = 00:00:00, threshold > 0.35)
```

### Stage 5: Ranking (3-Signal Hybrid)
```
final_score = 0.50 × chunk_similarity + 0.30 × title_similarity + 0.20 × keyword_score
Sort by final_score (primary), view_count (tiebreaker)
→ Return Top 3
```

### Stage 6: Relevance Check
```
If best final_score ≥ 0.45 → Return YouTube results
If best final_score < 0.45 → Fallback to web search (DuckDuckGo)
```

### Stage 7: Cache + Return
```
Store results in semantic cache → Display to user
```

---

## 📂 Project Structure

```
Distributed_Computing_Project/
│
├── main.py                         # CLI entry point (interactive + single query)
│
├── config/
│   ├── __init__.py
│   └── channels.py                 # YouTube channel IDs, subjects, all constants
│                                   # DASK_SCHEDULER_ADDRESS configured here
│
├── agents/
│   ├── __init__.py
│   ├── query_agent.py              # Query cleaning + multilingual embedding
│   ├── cache_agent.py              # SQLite semantic cache (store/search)
│   ├── router_agent.py             # Topic classification → subjects
│   ├── youtube_agent.py            # Video fetching + transcript/chapter processing
│   │                               # process_channel() runs on Dask workers
│   ├── ranking_agent.py            # 3-signal hybrid ranking
│   ├── web_agent.py                # DuckDuckGo fallback search
│   └── orchestrator.py             # Pipeline controller (Dask Client)
│
├── utils/
│   ├── __init__.py
│   ├── embeddings.py               # Multilingual MiniLM embeddings (384-dim)
│   ├── similarity.py               # Cosine similarity computation
│   └── cleaning.py                 # Text cleaning + chunking with overlap
│
├── db/
│   └── cache.db                    # SQLite cache (auto-created on first run)
│
├── requirements.txt                # All Python dependencies
└── README.md                       # This file
```

---

## 🚀 Setup Instructions (Single Machine)

### Prerequisites

- **Python 3.10+** (tested with 3.14)
- **pip** (package manager)
- **Git** (to clone the repository)
- **Internet connection** (for YouTube access and model download)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Distributed_Computing_Project
```

### Step 2: Create Virtual Environment

```bash
python -m venv myenv

# Activate it:
source myenv/bin/activate        # macOS / Linux
# myenv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package | Purpose |
|---------|---------|
| `langchain`, `langchain-community` | Document loading, YouTube transcripts |
| `yt-dlp` | YouTube video metadata & chapter extraction |
| `youtube-transcript-api` | Transcript fetching (backup method) |
| `sentence-transformers` | Multilingual embeddings (MiniLM) |
| `torch` | PyTorch backend for embeddings |
| `dask[distributed]` | Distributed parallel computing |
| `duckduckgo-search`, `ddgs` | Web search fallback |
| `beautifulsoup4`, `lxml` | Web page scraping |
| `scikit-learn`, `numpy` | Cosine similarity computation |

> **⚠️ First run:** The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, ~470 MB) will be downloaded automatically. This is cached for subsequent runs.

### Step 4: Run the System

```bash
# Interactive mode (keeps asking for queries)
python main.py

# Single query via CLI
python main.py "What is integration by parts?"

# Verbose mode (full pipeline logs)
python main.py "quantum mechanics explained" --verbose

# JSON output
python main.py "organic chemistry reactions" --json

# Clear the semantic cache
python main.py --clear-cache
```

### Example Queries

```bash
# English
python main.py "basic python"
python main.py "what is nextjs"
python main.py "binary search tree explained"

# Hindi
python main.py "न्यूटन के गति के नियम समझाइए"

# Hinglish (code-mixed)
python main.py "binary search tree kaise banaye"
python main.py "integration by parts in hindi"
```

---

## 🖥️ Running on Multiple Machines (Lab Setup)

This is the **distributed mode** where the workload is split across multiple lab PCs.

### Architecture

```
Machine 1 (Your PC)                    Machine 2 (Lab PC)
┌──────────────────────┐               ┌──────────────────────┐
│ Dask Scheduler       │               │ Dask Worker          │
│ (tcp://IP:8786)      │◄─────────────▶│ Connects to scheduler│
│                      │               │ Executes:            │
│ + Client (main.py)   │               │   process_channel()  │
│   Submits tasks      │               └──────────────────────┘
│   Gathers results    │
│   Ranks & displays   │               Machine 3 (Lab PC)
│                      │               ┌──────────────────────┐
│                      │◄─────────────▶│ Dask Worker          │
└──────────────────────┘               │ Executes:            │
                                       │   process_channel()  │
                                       └──────────────────────┘
```

### Prerequisites (ALL Machines)

**Every machine must have the exact same code and Python environment.**

```bash
# On EACH lab machine:
git clone <repository-url>
cd Distributed_Computing_Project
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### Step 1: Find Your IP Address

On Machine 1 (your main PC):

```bash
# macOS
ifconfig | grep "inet " | grep -v 127.0.0.1

# Linux
hostname -I

# Windows
ipconfig
```

Note your IP address (e.g., `192.168.1.10`).

### Step 2: Start the Dask Scheduler (Machine 1)

```bash
cd Distributed_Computing_Project
source myenv/bin/activate
dask scheduler
```

You will see output like:
```
Scheduler at:  tcp://192.168.1.10:8786
Dashboard at:  http://192.168.1.10:8787
```

> **Tip:** Open `http://192.168.1.10:8787` in a browser to see the real-time **Dask Dashboard** — it visualizes which worker is processing which channel.

### Step 3: Start Dask Workers (Machines 2, 3, 4...)

On each worker/lab machine:

```bash
cd Distributed_Computing_Project
source myenv/bin/activate
dask worker tcp://192.168.1.10:8786
```

Replace `192.168.1.10` with your Machine 1's actual IP address.

You should see:
```
Worker registered with scheduler at tcp://192.168.1.10:8786
```

### Step 4: Configure the Project

On Machine 1, edit `config/channels.py`:

```python
# Change this line:
DASK_SCHEDULER_ADDRESS = None

# To:
DASK_SCHEDULER_ADDRESS = "tcp://192.168.1.10:8786"
```

### Step 5: Run the Search (Machine 1)

```bash
python main.py "what is nextjs"
```

The system will:
1. Connect to the Dask scheduler (`tcp://192.168.1.10:8786`)
2. Submit `process_channel()` tasks for each YouTube channel
3. The scheduler distributes tasks to available workers
4. Workers on different machines execute in parallel
5. Results flow back to your machine for ranking and display

### What Happens on Each Worker

```
Worker receives: process_channel(channel_dict, query, query_embedding)
    │
    ├── 1. Fetch ALL videos from the channel (yt-dlp)
    ├── 2. Embed title+description of ALL videos
    ├── 3. Score against query embedding
    ├── 4. Filter: keep only videos with quick_score ≥ 0.25
    ├── 5. Select top-20 candidates
    ├── 6. Deep process each candidate:
    │       ├── Try transcript (Tier 1)
    │       ├── Try chapters (Tier 2)
    │       └── Try title match (Tier 3)
    └── 7. Return results with similarity + title_similarity + keyword_score
```

### Quick Summary

| Machine | Role | Command |
|---------|------|---------|
| Machine 1 (Your PC) | Scheduler + Client | `dask scheduler` then `python main.py` |
| Machine 2 (Lab PC) | Worker | `dask worker tcp://MACHINE_1_IP:8786` |
| Machine 3 (Lab PC) | Worker | `dask worker tcp://MACHINE_1_IP:8786` |
| Machine N... | Worker | `dask worker tcp://MACHINE_1_IP:8786` |

---

## 📖 Usage

### Interactive Mode

```bash
python main.py
```

```
=== Distributed Educational Search System ===
Type 'exit' to quit

Enter query: basic python
```

### CLI Mode

```bash
python main.py "basic python"
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--verbose` or `-v` | Show full pipeline logs (all stages) |
| `--json` | Output results as JSON |
| `--clear-cache` | Clear the semantic cache |

---

## 🧩 Module Descriptions

### 1. Query Agent (`agents/query_agent.py`)

| | |
|-|-|
| **Input** | Raw user query (any language) |
| **Output** | Cleaned query string + 384-dimensional embedding vector |
| **Model** | `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages) |

Handles Hindi (Devanagari), English, and Hinglish text. Cleans noise, normalizes whitespace, and generates a semantic embedding for downstream similarity computations.

### 2. Cache Agent (`agents/cache_agent.py`)

| | |
|-|-|
| **Backend** | SQLite (`db/cache.db`, auto-created) |
| **Strategy** | Semantic similarity (not exact match) |
| **Threshold** | Cosine similarity ≥ 0.89 = cache hit |

Stores query embeddings as BLOBs alongside JSON results. On new queries, computes cosine similarity against all cached embeddings. Dramatically reduces processing time for repeated/similar queries.

### 3. Router Agent (`agents/router_agent.py`)

| | |
|-|-|
| **Input** | Cleaned query + embedding |
| **Output** | List of subjects: `["computer_science"]`, `["mathematics", "physics"]`, etc. |
| **Strategy** | Hybrid: keyword matching (English + Hindi) → embedding similarity fallback |

If confidence is low, routes to ALL subjects (the ranking agent filters irrelevant results downstream).

### 4. YouTube Agent (`agents/youtube_agent.py`)

This is the core retrieval module. It runs on **Dask workers**.

**Two-Stage Pipeline:**

| Stage | What it Does |
|-------|-------------|
| **Stage 1: Metadata Filtering** | Fetch ALL channel videos → embed title+desc → score vs query → filter (≥ 0.25) → top-20 |
| **Stage 2: Deep Processing** | For each candidate, try 3 tiers in order: |

**3-Tier Deep Processing:**

| Tier | Method | Precision | When Used |
|------|--------|-----------|-----------|
| **Tier 1** | Transcript → chunk → embed → match | **Precise timestamps** (e.g., 00:05:32) | Default — loads full transcript |
| **Tier 2** | Chapter markers → embed titles → match | **Accurate timestamps** (chapter start) | When YouTube blocks transcript (IP blocked) |
| **Tier 3** | Title + description match | **No timestamp** (00:00:00) | Last resort, threshold > 0.35 |

**Key Feature:** Fetches **ALL videos** from each channel (no `playlistend` limit), then uses metadata filtering to select the most relevant candidates. This ensures old but relevant videos (e.g., a 2021 Next.js tutorial) are not missed.

### 5. Ranking Agent (`agents/ranking_agent.py`)

**3-Signal Hybrid Scoring Model:**

```
final_score = 0.50 × chunk_similarity      (deep semantic match)
            + 0.30 × title_similarity      (metadata intent match)
            + 0.20 × keyword_score         (literal keyword overlap)
```

| Step | Action |
|------|--------|
| **Deduplicate** | Keep best chunk per video |
| **Score** | Compute hybrid `final_score` using 3 signals |
| **Sort** | By `final_score` (primary) + `view_count` (tiebreaker) |
| **Select** | Top 3 results |
| **Fallback check** | If best `final_score` < 0.45 → trigger web fallback |

> **Design Decision:** View count is a **tiebreaker**, NOT the primary ranking signal. This prevents viral-but-irrelevant videos from dominating results.

### 6. Web Agent (`agents/web_agent.py`)

| | |
|-|-|
| **When** | Activated when no relevant YouTube results found |
| **Search** | DuckDuckGo (free, no API key) |
| **Process** | Load top 5 pages → chunk → embed → rank |
| **Boost** | Trusted educational sites get 1.15× score multiplier |

**Trusted Sites:** Wikipedia, Khan Academy, GeeksforGeeks, Brilliant, NCERT, StackOverflow, etc.

### 7. Orchestrator (`agents/orchestrator.py`)

The **master controller** that connects all agents into a pipeline.

| | |
|-|-|
| **Dask Client** | Persistent connection to scheduler (created once in `__init__`) |
| **Task Submission** | `client.submit(process_channel, ...)` per channel |
| **Result Collection** | `client.gather(futures)` — blocks until all workers finish |
| **Fallback** | If distributed mode fails → falls back to local threaded execution |

---

## 📊 Ranking Algorithm

### Why 3-Signal Scoring?

| Signal | What it Measures | Weight | Why |
|--------|-----------------|--------|-----|
| `chunk_similarity` | How well the transcript/chapter content matches the query | 0.50 | Deep content relevance |
| `title_similarity` | How well the video title/description matches the query | 0.30 | Topic intent match |
| `keyword_score` | Fraction of query words found literally in title+description | 0.20 | Catches exact term matches |
| `view_count` | Video popularity | Tiebreaker | Prevents obscure duplicates |

### Filtering at Every Stage

```
ALL videos (e.g., 500)
    │
    ├── Metadata filter (quick_score ≥ 0.25)  → ~50 pass
    │
    ├── Top-K selection                        → 20 selected
    │
    ├── Transcript chunks (similarity ≥ 0.30)  → weak chunks dropped
    │   OR Chapter matches (similarity ≥ 0.30)
    │   OR Title fallback (similarity > 0.35)
    │
    ├── Deduplicate (best chunk per video)
    │
    ├── Hybrid scoring + sort
    │
    └── Top 3 returned (if final_score ≥ 0.45)
```

---

## 📡 Distributed Nodes (Subjects)

Each subject maps to a set of predefined YouTube channels:

| Node | Subject | Channels | Language |
|------|---------|----------|----------|
| 1 | **Mathematics** | 3Blue1Brown, MIT OpenCourseWare, Khan Academy, Vedantu JEE, Physics Wallah | EN + HI |
| 2 | **Computer Science** | freeCodeCamp, CodeWithHarry, NPTEL | EN + HI |
| 3 | **Physics** | Physics Wallah, MIT OpenCourseWare, Vedantu JEE | EN + HI |
| 4 | **Chemistry** | Khan Academy, The Organic Chemistry Tutor, Vedantu JEE, Physics Wallah | EN + HI |

> **Adding channels:** Edit `config/channels.py` → add a new dict with `name`, `channel_id`, and `language` to the appropriate list.

To find a channel's ID:
1. Go to the YouTube channel page
2. View page source → search for `channel_id`
3. Or use: `https://www.youtube.com/channel/CHANNEL_ID`

---

## 🔧 Configuration Reference

All configurable parameters are in `config/channels.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DASK_SCHEDULER_ADDRESS` | `None` | Scheduler address for distributed mode. Set to `"tcp://IP:8786"` for multi-machine. |
| `MAX_VIDEOS_PER_CHANNEL` | `15` | Reference value (actual fetching has NO limit) |
| `MAX_DASK_WORKERS` | `8` | Max parallel threads (local mode only) |
| `CACHE_SIMILARITY_THRESHOLD` | `0.89` | Cosine similarity threshold for cache hits |
| `TOP_SIMILARITY_RESULTS` | `10` | Results kept after hybrid scoring |
| `TOP_FINAL_RESULTS` | `3` | Final number of results returned to user |
| `WEB_SEARCH_MAX_URLS` | `5` | Max URLs for web fallback |
| `TRANSCRIPT_CHUNK_SIZE` | `500` | Characters per transcript chunk (~30 sec of speech) |
| `TRANSCRIPT_CHUNK_OVERLAP` | `50` | Overlap between chunks (avoids cutting sentences) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embedding model |
| `TRUSTED_SITE_BOOST` | `1.15` | Score multiplier for trusted educational websites |

### Tuning for Better Results

| Issue | Adjustment |
|-------|-----------|
| Too many irrelevant results | Increase `MIN_QUICK_SCORE` in `youtube_agent.py` (default: 0.25) |
| Missing relevant old videos | Already fetches ALL videos. Increase `TOP_K` in `youtube_agent.py` (default: 20) |
| Too aggressive web fallback | Lower `min_similarity` in `ranking_agent.py` (default: 0.45) |
| "No results found" too often | Lower `MIN_QUICK_SCORE` to 0.20 or `min_similarity` to 0.35 |

---

## 📊 Output Format

### YouTube Results

```
==============================
RESULT
==============================

Query: basic python
Source: youtube
Time: 136.31s

[1] Python for Everybody - Full University Python Course
Channel: freeCodeCamp
Timestamp: 00:00:00
Link: https://www.youtube.com/watch?v=8DvywoWv6fI
Score: 0.6872  |  Final: 0.7498
Views: 7643393
Snippet: [Matched by title] Python for Everybody - Full University Python Course
--------------------------------------------------
[2] Intermediate Python Programming Course
Channel: freeCodeCamp
Timestamp: 00:00:00
Link: https://www.youtube.com/watch?v=HGOBQPFzWKo
Score: 0.6163  |  Final: 0.693
Views: 4349596
Snippet: [Matched by title] Intermediate Python Programming Course
--------------------------------------------------
```

| Field | Meaning |
|-------|---------|
| `Score` | Raw `chunk_similarity` (semantic match of transcript/title) |
| `Final` | Hybrid `final_score` (0.50×chunk + 0.30×title + 0.20×keyword) |
| `Timestamp` | Precise timestamp in the video where the topic is discussed |
| `Link` | Direct YouTube link with timestamp (`&t=XXs`) |

### Web Results (Fallback)

```
[1] What Is Next.js? Framework Guide & Why Use It in 2026
URL: https://pagepro.co/blog/what-is-nextjs/
Score: 0.7682
Snippet: Next.js is a React framework that adds features like server-side rendering...
--------------------------------------------------
```

### Cache Results

When a similar query was previously searched:

```
Query: basic python
Source: cache
Cached query: basic python
Similarity: 0.9998
(Previously cached results displayed)
```

---

## ❗ Troubleshooting

### YouTube IP Blocked

```
ERROR: Sign in to confirm you're not a bot.
YouTubeAgent: ⚠️ YouTube IP BLOCKED — switching to chapter-based matching.
```

**Cause:** YouTube rate-limits IPs that make too many requests.

**Solutions:**
1. **Wait 15-30 minutes** — the block usually expires
2. Use **cookies** for authentication:
   ```bash
   # Export cookies from your browser
   yt-dlp --cookies-from-browser chrome --write-cookies cookies.txt
   ```
3. The system **automatically falls back** to chapter-based matching (Tier 2) or title matching (Tier 3) when blocked. Transcripts won't load, but results still work.

### Slow First Run

**Cause:** The embedding model (~470 MB) is being downloaded.

**Solution:** This only happens once. The model is cached in `~/.cache/huggingface/`.

### "No videos passed metadata filter"

**Cause:** The query doesn't match any video titles/descriptions in the channel.

**Solutions:**
1. This is expected — some channels just don't have content on that topic
2. The system searches multiple channels and falls back to web search
3. To be more lenient: lower `MIN_QUICK_SCORE` in `youtube_agent.py` from 0.25 to 0.20

### Dask Worker Connection Failed

```
Orchestrator: Failed to connect to Dask scheduler
```

**Solutions:**
1. Check that `dask scheduler` is running on Machine 1
2. Check that the IP address in `DASK_SCHEDULER_ADDRESS` is correct
3. Ensure all machines are on the **same network** (same WiFi/LAN)
4. Check firewall: port `8786` must be open
5. The system will **automatically fall back to local mode** if connection fails

### Too Many Results / Slow Performance

**Cause:** Fetching ALL videos from channels with 5000+ videos takes time.

**Solutions:**
1. This is intentional — ensures old relevant videos are found
2. The metadata filtering stage prunes quickly — only top-20 go to deep processing
3. Use **distributed mode** to parallelize across machines

---

## 📝 Important Notes

- **No paid APIs** — Everything uses free, open-source tools
- **CPU-only** — Designed to run without GPU; sentence-transformers works well on CPU
- **Predefined channels only** — The system NEVER searches global YouTube. All videos come from configured channels in `config/channels.py`
- **Rate limiting** — The system uses caching and metadata filtering to minimize YouTube API calls
- **Transcript availability** — Not all videos have transcripts. The 3-tier fallback handles this gracefully
- **First run is slow** — Model download + cold start. Subsequent runs are much faster (especially with cache hits)

---

## 🧪 Evaluation

The system provides evaluation scores for transparency:

| Metric | Meaning |
|--------|---------|
| `mean_similarity` | Average raw cosine similarity of returned results |
| `max_similarity` | Best raw match score |
| `min_similarity` | Lowest raw match in the returned set |
| `mean_final_score` | Average hybrid score |
| `max_final_score` | Best hybrid score |

View these with `--json` flag:

```bash
python main.py "what is python" --json
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Distributed Computing | Dask Distributed |
| Embeddings | sentence-transformers (MiniLM L12 v2) |
| YouTube | yt-dlp, youtube-transcript-api |
| Web Search | DuckDuckGo (via ddgs) |
| Web Scraping | LangChain WebBaseLoader, BeautifulSoup4 |
| Similarity | scikit-learn (cosine similarity) |
| Cache | SQLite3 (built-in) |
| ML Backend | PyTorch |

---

## 📜 License

This project is for educational purposes. Built as part of a Distributed Computing course project.
