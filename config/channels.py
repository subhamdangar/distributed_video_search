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
    {
        "name": "Vedantu JEE Made Ejee",
        "channel_id": "UC91RZv71f8p0VV2gaFI07pg",
        "language": "hi",
    },
    {
        "name": "Physics Wallah - Alakh Pandey",
        "channel_id": "UCiGyWN6DEbnj2alu7iapuKQ",
        "language": "hi",
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
        "name": "Physics Wallah - Alakh Pandey",
        "channel_id": "UCiGyWN6DEbnj2alu7iapuKQ",
        "language": "hi",
    },
    {
        "name": "MIT OpenCourseWare",
        "channel_id": "UCEBb1b_L6zDS3xTUrIALZOw",
        "language": "en",
    },
    {
        "name": "Vedantu JEE Made Ejee",
        "channel_id": "UC91RZv71f8p0VV2gaFI07pg",
        "language": "hi",
    },
]

# ──────────────────────────────────────────────────────────────
# Node 4: Chemistry
# ──────────────────────────────────────────────────────────────
CHEMISTRY_CHANNELS = [

    {
        "name": "Khan Academy",
        "channel_id": "UC4a-Gbdw7vOaccHmFo40b9g",
        "language": "en",
    },
    {
        "name": "The Organic Chemistry Tutor",
        "channel_id": "UCEWpbFLzoYGPfuWUMFPSaoA",
        "language": "en",
    },
    {
        "name": "Vedantu JEE Made Ejee",
        "channel_id": "UC91RZv71f8p0VV2gaFI07pg",
        "language": "hi",
    },
    {
        "name": "Physics Wallah - Alakh Pandey",
        "channel_id": "UCiGyWN6DEbnj2alu7iapuKQ",
        "language": "hi",
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

# Maximum videos per channel (used as a reference; actual fetching uses NO limit)
# The metadata filtering stage selects top-K from ALL fetched videos.
MAX_VIDEOS_PER_CHANNEL = 15

# Maximum Dask workers for parallel execution (local mode only)
MAX_DASK_WORKERS = 8

# ──────────────────────────────────────────────────────────────
# Dask Distributed Configuration
# ──────────────────────────────────────────────────────────────
# Set to the scheduler address for multi-machine distributed execution.
# Set to None for local-only mode (threads on this machine).
#
# HOW TO USE IN A LAB:
#
#   Machine 1 (Scheduler + Client):
#     $ dask scheduler
#     → Scheduler running at tcp://MACHINE_1_IP:8786
#     → Dashboard at http://MACHINE_1_IP:8787
#
#   Machine 2, 3, 4... (Workers):
#     $ dask worker tcp://MACHINE_1_IP:8786
#
#   Then set:
#     DASK_SCHEDULER_ADDRESS = "tcp://MACHINE_1_IP:8786"
#
# The client (main.py) ALSO runs on Machine 1.
# ──────────────────────────────────────────────────────────────
DASK_SCHEDULER_ADDRESS = None

# Semantic cache similarity threshold (cosine)
# Queries with similarity >= this value are considered cache hits
CACHE_SIMILARITY_THRESHOLD = 0.89

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
