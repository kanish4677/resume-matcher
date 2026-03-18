"""
Resume / Job Description Matcher
Core ML model using sentence-transformers + cosine similarity
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    overall_score: float          # 0–100
    semantic_score: float         # cosine sim via embeddings
    keyword_score: float          # TF-IDF keyword overlap
    matched_keywords: List[str]   # keywords present in both
    missing_keywords: List[str]   # keywords in JD but absent in resume
    section_scores: dict          # per-section breakdown
    recommendations: List[str]    # actionable suggestions

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        grade = (
            "Excellent" if self.overall_score >= 80 else
            "Good"      if self.overall_score >= 60 else
            "Fair"      if self.overall_score >= 40 else
            "Poor"
        )
        return (
            f"Match: {grade} ({self.overall_score:.1f}/100)\n"
            f"  Semantic similarity : {self.semantic_score:.1f}/100\n"
            f"  Keyword overlap     : {self.keyword_score:.1f}/100\n"
            f"  Matched keywords    : {', '.join(self.matched_keywords[:8]) or 'none'}\n"
            f"  Missing keywords    : {', '.join(self.missing_keywords[:8]) or 'none'}\n"
        )


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall",
    "i","you","he","she","we","they","it","this","that","these","those",
    "my","your","his","her","our","their","its","which","who","whom","when",
    "where","why","how","all","each","every","both","few","more","most",
    "other","such","than","then","so","yet","both","whether","about",
    "as","if","while","although","because","since","unless","until","after",
    "before","during","through","between","into","over","under","above",
    "below","within","without","throughout","regarding","per","via",
}

TECH_SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "dl": "deep learning",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "pytorch": "torch",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "ci/cd": "continuous integration continuous deployment",
    "db": "database",
    "sql": "structured query language",
    "api": "application programming interface",
    "oop": "object oriented programming",
    "rest": "representational state transfer",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
}


def preprocess(text: str, expand_synonyms: bool = True) -> str:
    """Lowercase, remove special chars, expand common tech abbreviations."""
    text = text.lower()
    text = re.sub(r"[^\w\s\+\#\/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if expand_synonyms:
        for abbr, full in TECH_SYNONYMS.items():
            text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text


def extract_keywords(text: str, top_n: int = 40) -> List[str]:
    """Extract top TF-IDF keywords from a single document."""
    processed = preprocess(text)
    tokens = processed.split()
    # Remove stop words and short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    # Use character n-grams + word n-grams for tech terms like 'scikit-learn'
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=200,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([" ".join(tokens)])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in ranked[:top_n]]
    except ValueError:
        return tokens[:top_n]


def extract_sections(text: str) -> dict:
    """
    Heuristically split resume/JD into logical sections.
    Returns dict of section_name -> text.
    """
    section_patterns = {
        "skills":       r"(?:technical\s+)?skills?|technologies|tech\s+stack|tools",
        "experience":   r"experience|employment|work\s+history|professional\s+background",
        "education":    r"education|academic|degree|university|college",
        "projects":     r"projects?|portfolio|side\s+projects?|personal\s+projects?",
        "certifications": r"certifi(?:cation|ed)|licenses?|credentials?",
        "summary":      r"summary|objective|profile|about\s+me|overview",
    }
    lines = text.split("\n")
    sections: dict = {"full": text}
    current_section = "other"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        matched_section = None
        for sec, pattern in section_patterns.items():
            if re.match(rf"^(?:{pattern})\s*[:\-]?\s*$", stripped, re.IGNORECASE):
                matched_section = sec
                break
        if matched_section:
            if current_lines:
                sections[current_section] = " ".join(current_lines)
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(stripped)

    if current_lines:
        sections[current_section] = " ".join(current_lines)
    return sections


# ---------------------------------------------------------------------------
# Embedding backend (pluggable)
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    """
    Fallback embedder using TF-IDF vectors.
    Swap this out for SentenceTransformer for production quality.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english",
        )
        self._fitted = False

    def encode(self, texts: List[str]) -> np.ndarray:
        processed = [preprocess(t) for t in texts]
        if not self._fitted:
            self.vectorizer.fit(processed)
            self._fitted = True
        matrix = self.vectorizer.transform(processed)
        return matrix.toarray()


try:
    from sentence_transformers import SentenceTransformer

    class STEmbedder:
        """Production embedder using all-MiniLM-L6-v2 (~80MB, very fast)."""
        MODEL_NAME = "all-MiniLM-L6-v2"

        def __init__(self):
            print(f"Loading SentenceTransformer: {self.MODEL_NAME}")
            self.model = SentenceTransformer(self.MODEL_NAME)

        def encode(self, texts: List[str]) -> np.ndarray:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    DEFAULT_EMBEDDER = STEmbedder
    EMBEDDER_NAME = "SentenceTransformer (all-MiniLM-L6-v2)"

except ImportError:
    DEFAULT_EMBEDDER = TFIDFEmbedder
    EMBEDDER_NAME = "TF-IDF (fallback — install sentence-transformers for better results)"


# ---------------------------------------------------------------------------
# Core Matcher
# ---------------------------------------------------------------------------

class ResumeMatcher:
    """
    Main matcher class.

    Usage:
        matcher = ResumeMatcher()
        result  = matcher.match(resume_text, job_description_text)
        print(result.summary())
    """

    # Weight for blending semantic + keyword scores
    SEMANTIC_WEIGHT = 0.65
    KEYWORD_WEIGHT  = 0.35

    def __init__(self, embedder=None):
        self.embedder = embedder or DEFAULT_EMBEDDER()
        print(f"ResumeMatcher ready — using {EMBEDDER_NAME}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, resume: str, job_description: str) -> MatchResult:
        """Score a resume against a job description. Returns MatchResult."""
        resume_sections = extract_sections(resume)
        jd_sections     = extract_sections(job_description)

        # 1. Semantic score (full text)
        semantic = self._semantic_score(resume, job_description)

        # 2. Keyword score
        resume_kws = set(extract_keywords(resume, top_n=60))
        jd_kws     = set(extract_keywords(job_description, top_n=60))
        matched    = sorted(resume_kws & jd_kws)
        missing    = sorted(jd_kws - resume_kws)
        kw_score   = (len(matched) / len(jd_kws) * 100) if jd_kws else 0.0

        # 3. Per-section scores
        section_scores = self._section_scores(resume_sections, jd_sections)

        # 4. Weighted overall
        overall = (
            self.SEMANTIC_WEIGHT * semantic +
            self.KEYWORD_WEIGHT  * kw_score
        )
        overall = round(min(overall, 100), 1)

        # 5. Recommendations
        recs = self._recommendations(overall, matched, missing, section_scores)

        return MatchResult(
            overall_score    = overall,
            semantic_score   = round(semantic, 1),
            keyword_score    = round(kw_score, 1),
            matched_keywords = matched[:15],
            missing_keywords = missing[:15],
            section_scores   = section_scores,
            recommendations  = recs,
        )

    def rank(
        self,
        resumes: List[Tuple[str, str]],   # list of (name, resume_text)
        job_description: str,
    ) -> List[Tuple[str, MatchResult]]:
        """Rank multiple resumes against one JD. Returns sorted list."""
        results = [(name, self.match(text, job_description)) for name, text in resumes]
        return sorted(results, key=lambda x: x[1].overall_score, reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _semantic_score(self, text_a: str, text_b: str) -> float:
        embeddings = self.embedder.encode([preprocess(text_a), preprocess(text_b)])
        sim = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
        return float(sim) * 100

    def _section_scores(self, resume_secs: dict, jd_secs: dict) -> dict:
        scores = {}
        shared_sections = set(resume_secs) & set(jd_secs) - {"full"}
        for sec in shared_sections:
            score = self._semantic_score(resume_secs[sec], jd_secs[sec])
            scores[sec] = round(score, 1)
        return scores

    def _recommendations(
        self,
        overall: float,
        matched: List[str],
        missing: List[str],
        section_scores: dict,
    ) -> List[str]:
        recs = []

        if missing:
            top_missing = ", ".join(missing[:5])
            recs.append(
                f"Add missing keywords to your resume: {top_missing}. "
                "Include them naturally in your experience bullet points."
            )

        if overall < 50:
            recs.append(
                "Your resume has low overall alignment. Consider tailoring it "
                "specifically for this role — rewrite your summary and skills section "
                "to mirror the job description language."
            )
        elif overall < 70:
            recs.append(
                "Good foundation. Strengthen weak sections by quantifying achievements "
                "and using the exact terminology from the job description."
            )
        else:
            recs.append(
                "Strong match! Make sure your top 3 relevant projects are prominently "
                "listed and your summary opens with role-specific language."
            )

        weak_sections = [s for s, score in section_scores.items() if score < 50]
        if weak_sections:
            recs.append(
                f"Weak section alignment: {', '.join(weak_sections)}. "
                "Expand these sections to better reflect the job requirements."
            )

        if len(matched) < 10:
            recs.append(
                "Low keyword density. Add a dedicated 'Technical Skills' section "
                "listing all relevant tools, languages, and frameworks."
            )

        return recs
