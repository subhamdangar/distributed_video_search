"""
Router Agent
────────────
Classifies a user query into one of the subject categories:
  - mathematics
  - computer_science
  - physics
  - chemistry

Uses a hybrid approach:
  1. Keyword matching (fast, handles obvious cases)
  2. Embedding similarity against subject descriptions (handles ambiguous/Hindi queries)

Falls back to returning ALL subjects if confidence is low (let the
ranking agent figure it out).
"""

import logging
import numpy as np
from utils.embeddings import embed_text, embed_texts
from utils.similarity import compute_similarity

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Keyword dictionaries for each subject (English + Hindi)
# ──────────────────────────────────────────────────────────────
SUBJECT_KEYWORDS = {
    "mathematics": [
        "math", "maths", "mathematics", "algebra", "calculus", "geometry",
        "trigonometry", "linear algebra", "probability", "statistics",
        "differential", "integral", "matrix", "matrices", "equation",
        "theorem", "proof", "number theory", "topology",
        "गणित", "बीजगणित", "कैलकुलस", "ज्यामिति", "त्रिकोणमिति",
        "समीकरण", "प्रमेय", "सांख्यिकी",
    ],
    "computer_science": [
        "computer", "programming", "coding", "algorithm", "data structure",
        "python", "java", "javascript", "machine learning", "deep learning",
        "artificial intelligence", "ai", "ml", "neural network", "database",
        "sql", "web development", "software", "operating system", "compiler",
        "dsa", "leetcode", "competitive programming", "oop",
        "binary", "search tree", "linked list", "stack", "queue", "hash",
        "graph", "sort", "sorting", "array", "recursion", "loop", "pointer",
        "heap", "dynamic programming", "backtracking", "bfs", "dfs",
        "api", "frontend", "backend", "html", "css", "react", "node",
        "कंप्यूटर", "प्रोग्रामिंग", "कोडिंग", "एल्गोरिदम",
        "मशीन लर्निंग", "डीप लर्निंग", "सॉफ्टवेयर",
    ],
    "physics": [
        "physics", "mechanics", "thermodynamics", "electrodynamics",
        "quantum", "relativity", "optics", "waves", "electromagnetism",
        "newton", "force", "energy", "momentum", "velocity", "acceleration",
        "nuclear", "particle", "astrophysics", "gravity", "electric field",
        "magnetic field", "circuit",
        "भौतिकी", "भौतिक", "यांत्रिकी", "ऊष्मागतिकी", "प्रकाशिकी",
        "गुरुत्वाकर्षण", "बल", "ऊर्जा", "विद्युत", "चुंबकीय",
    ],
    "chemistry": [
        "chemistry", "chemical", "organic", "inorganic", "physical chemistry",
        "molecule", "atom", "bond", "reaction", "acid", "base", "pH",
        "periodic table", "element", "compound", "solution", "molar",
        "electrochemistry", "polymer", "catalyst",
        "रसायन", "रसायनशास्त्र", "अणु", "परमाणु", "अम्ल", "क्षार",
        "आवर्त सारणी", "तत्व", "यौगिक", "विलयन",
    ],
}

# ──────────────────────────────────────────────────────────────
# Rich subject descriptions for embedding-based classification
# ──────────────────────────────────────────────────────────────
SUBJECT_DESCRIPTIONS = {
    "mathematics": (
        "Mathematics including algebra, calculus, geometry, trigonometry, "
        "linear algebra, probability, statistics, number theory, differential "
        "equations, mathematical proofs and theorems. "
        "गणित, बीजगणित, कैलकुलस, ज्यामिति, त्रिकोणमिति"
    ),
    "computer_science": (
        "Computer science including programming, algorithms, data structures, "
        "machine learning, artificial intelligence, web development, databases, "
        "software engineering, operating systems, and coding tutorials. "
        "कंप्यूटर विज्ञान, प्रोग्रामिंग, कोडिंग, एल्गोरिदम"
    ),
    "physics": (
        "Physics including mechanics, thermodynamics, electromagnetism, optics, "
        "quantum physics, relativity, astrophysics, nuclear physics, waves, "
        "forces, energy, and motion. "
        "भौतिकी, यांत्रिकी, ऊष्मागतिकी, प्रकाशिकी, क्वांटम"
    ),
    "chemistry": (
        "Chemistry including organic chemistry, inorganic chemistry, physical "
        "chemistry, chemical reactions, periodic table, molecular structure, "
        "electrochemistry, acids, bases, and polymers. "
        "रसायन विज्ञान, रसायनशास्त्र, कार्बनिक, अकार्बनिक"
    ),
}


class RouterAgent:
    """
    Routes a query to one or more subjects using keyword matching
    and embedding-based similarity.
    """

    def __init__(self):
        # Pre-compute subject description embeddings (done once)
        self._subject_names = list(SUBJECT_DESCRIPTIONS.keys())
        self._subject_embeddings = None  # Lazy-loaded

    def _ensure_embeddings(self):
        """Lazy-load subject description embeddings."""
        if self._subject_embeddings is None:
            descriptions = [SUBJECT_DESCRIPTIONS[s] for s in self._subject_names]
            self._subject_embeddings = embed_texts(descriptions)
            logger.info("RouterAgent: Subject embeddings computed.")

    def route(self, cleaned_query: str, query_embedding: np.ndarray) -> list[str]:
        """
        Determine which subject(s) a query belongs to.

        Strategy:
          1. Check keywords → if exact match found, return that subject
          2. Use embedding similarity → pick best subject if confident
          3. If unsure → return all subjects (let ranking sort it out)

        Args:
            cleaned_query: The cleaned query string.
            query_embedding: The query embedding vector.

        Returns:
            List of subject names (e.g., ['physics'] or ['mathematics', 'physics']).
        """
        query_lower = cleaned_query.lower()

        # ── Step 1: Keyword matching ──────────────────────────
        keyword_hits = {}
        for subject, keywords in SUBJECT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                keyword_hits[subject] = score

        if keyword_hits:
            # Sort by match count, take top subject(s)
            sorted_subjects = sorted(keyword_hits.items(), key=lambda x: x[1], reverse=True)
            best_score = sorted_subjects[0][1]

            # If clear winner (2x the second), return just that subject
            if len(sorted_subjects) == 1 or sorted_subjects[0][1] >= 2 * sorted_subjects[1][1]:
                result = [sorted_subjects[0][0]]
                logger.info(f"RouterAgent: Keyword match → {result}")
                return result

            # Multiple close matches — return top 2
            result = [s[0] for s in sorted_subjects[:2]]
            logger.info(f"RouterAgent: Multiple keyword matches → {result}")
            return result

        # ── Step 2: Embedding similarity ──────────────────────
        self._ensure_embeddings()
        similarities = compute_similarity(query_embedding, self._subject_embeddings)

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        logger.info(f"RouterAgent: Embedding similarities = "
                     f"{dict(zip(self._subject_names, similarities.round(4)))}")

        # If confident enough, return the best match
        if best_score >= 0.40:
            result = [self._subject_names[best_idx]]
            logger.info(f"RouterAgent: Embedding match (score={best_score:.4f}) → {result}")
            return result

        # ── Step 3: Low confidence — return all subjects ──────
        logger.info("RouterAgent: Low confidence — routing to ALL subjects.")
        return list(self._subject_names)
