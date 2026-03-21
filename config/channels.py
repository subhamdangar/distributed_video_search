"""
Channel Configuration for the Distributed Multilingual Educational Search System.

Each subject (node) has a list of predefined YouTube channels.
Channels are stored as dictionaries with:
  - name: Human-readable channel name
  - channel_id: YouTube channel ID (used for constructing URLs and fetching videos)
  - language: Primary language of the channel (en / hi / mixed)

To add/remove channels, simply edit the lists below.
"""

# ──────────────────────────────────────────────────────────────
# Node 1: Mathematics
# ──────────────────────────────────────────────────────────────
MATHEMATICS_CHANNELS = [
    {
        "name": "3Blue1Brown",
        "channel_id": "UCYO_jab_esuFRV4b17AJtAw",
        "language": "en",
    },
    {
        "name": "MIT OpenCourseWare",
        "channel_id": "UCEBb1b_L6zDS3xTUrIALZOw",
        "language": "en",
    },
    {
        "name": "Khan Academy",
        "channel_id": "UC4a-Gbdw7vOaccHmFo40b9g",
        "language": "en",
    },
]

# ──────────────────────────────────────────────────────────────
# Node 2: Computer Science
# ──────────────────────────────────────────────────────────────
COMPUTER_SCIENCE_CHANNELS = [
    {
        "name": "freeCodeCamp",
        "channel_id": "UC8butISFwT-Wl7EV0hUK0BQ",
        "language": "en",
    },
    {
        "name": "CodeWithHarry",
        "channel_id": "UCeVMnSShP_Iviwkknt83cww",
        "language": "hi",
    },
    {
        "name": "NPTEL",
        "channel_id": "UCnEknCMO52EU0d4wYR5CJDQ",
        "language": "en",
    },
]

# ──────────────────────────────────────────────────────────────
# Node 3: Physics
# ──────────────────────────────────────────────────────────────
PHYSICS_CHANNELS = [
    {
        "name": "Physics Wallah",
        "channel_id": "UCHOPs2hX3WCaVxVF4FmtSDg",
        "language": "hi",
    },
    {
        "name": "MIT OpenCourseWare",
        "channel_id": "UCEBb1b_L6zDS3xTUrIALZOw",
        "language": "en",
    },
]

# ──────────────────────────────────────────────────────────────
# Node 4: Chemistry
# ──────────────────────────────────────────────────────────────
CHEMISTRY_CHANNELS = [
    {
        "name": "Unacademy",
        "channel_id": "UCBGIijVBNZRXhj6XI7KBNlw",
        "language": "hi",
    },
    {
        "name": "Khan Academy",
        "channel_id": "UC4a-Gbdw7vOaccHmFo40b9g",
        "language": "en",
    },
]


# ──────────────────────────────────────────────────────────────
# Unified mapping: subject → channels
# This is used by the Router Agent to dispatch work
# ──────────────────────────────────────────────────────────────
SUBJECT_CHANNELS = {
    "mathematics": MATHEMATICS_CHANNELS,
    "computer_science": COMPUTER_SCIENCE_CHANNELS,
    "physics": PHYSICS_CHANNELS,
    "chemistry": CHEMISTRY_CHANNELS,
}

# ──────────────────────────────────────────────────────────────
# System-level configuration constants
# ──────────────────────────────────────────────────────────────

# Maximum number of recent videos to fetch per channel
MAX_VIDEOS_PER_CHANNEL = 15

# Maximum Dask workers for parallel execution
MAX_DASK_WORKERS = 8

# Semantic cache similarity threshold (cosine)
# Queries with similarity >= this value are considered cache hits
CACHE_SIMILARITY_THRESHOLD = 0.88

# Number of top results by similarity before ranking by views
TOP_SIMILARITY_RESULTS = 10

# Final number of results to return to user
TOP_FINAL_RESULTS = 3

# Web search fallback: max URLs to process
WEB_SEARCH_MAX_URLS = 5

# Chunk size (in characters) for transcript segmentation
# Roughly maps to ~30 seconds of speech
TRANSCRIPT_CHUNK_SIZE = 500

# Overlap between chunks to avoid cutting sentences
TRANSCRIPT_CHUNK_OVERLAP = 50

# Embedding model name (multilingual MiniLM supports 50+ languages)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# SQLite database path (relative to project root)
import os
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "cache.db")

# Trusted educational websites (get boosted ranking in web search)
TRUSTED_WEBSITES = [
    "wikipedia.org",
    "khan.academy",
    "brilliant.org",
    "geeksforgeeks.org",
    "mathworld.wolfram.com",
    "hyperphysics.phy-astr.gsu.edu",
    "ncert.nic.in",
    "byjus.com",
    "stackexchange.com",
    "stackoverflow.com",
]

# Trusted site boost factor (multiplied to similarity score)
TRUSTED_SITE_BOOST = 1.15
