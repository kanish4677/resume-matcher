"""
Tests for the Resume Matcher core logic.
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from src.matcher import (
    ResumeMatcher,
    preprocess,
    extract_keywords,
    extract_sections,
    MatchResult,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

RESUME_STRONG = """
Jane Doe — Machine Learning Engineer
SKILLS
Python, PyTorch, TensorFlow, scikit-learn, NLP, computer vision, Docker, FastAPI,
MLflow, AWS, SQL, Pandas, NumPy, LangChain, Hugging Face Transformers
EXPERIENCE
ML Engineer at TechCorp 2023-present
- Built BERT fine-tuned document classification pipeline, 50k docs/day
- Reduced inference latency 40% via ONNX quantization
- Deployed FastAPI model serving on AWS ECS with Docker
PROJECTS
RAG Q&A system using LangChain and ChromaDB. Crop disease detection with EfficientNet.
EDUCATION
B.Tech Computer Science, IIT, CGPA 8.6
"""

RESUME_WEAK = """
John Smith — Graphic Designer
I make beautiful logos and illustrations using Adobe Photoshop and Illustrator.
I am creative and love art. I have designed many websites using basic HTML.
"""

JD_ML = """
Senior ML Engineer position
Requirements:
- 2+ years machine learning experience
- Python, PyTorch or TensorFlow
- NLP experience with Hugging Face Transformers
- Model deployment: FastAPI, Docker, Kubernetes
- MLOps: MLflow, experiment tracking
- AWS or GCP cloud experience
- SQL, Pandas, NumPy for data manipulation
Nice to have: LangChain, RAG pipelines, computer vision
"""


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_lowercase(self):
        assert preprocess("Python PyTorch") == "python pytorch"

    def test_remove_special_chars(self):
        result = preprocess("hello! world@2024")
        assert "!" not in result
        assert "@" not in result

    def test_synonym_expansion(self):
        result = preprocess("ml and nlp skills", expand_synonyms=True)
        assert "machine learning" in result
        assert "natural language processing" in result

    def test_no_synonym_expansion(self):
        result = preprocess("ml skills", expand_synonyms=False)
        assert "ml" in result
        assert "machine learning" not in result

    def test_whitespace_normalised(self):
        result = preprocess("  hello   world  ")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# Keyword extraction tests
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_returns_list(self):
        kws = extract_keywords("Python machine learning deep learning NLP")
        assert isinstance(kws, list)

    def test_top_n_respected(self):
        text = " ".join([f"word{i}" for i in range(100)])
        kws = extract_keywords(text, top_n=10)
        assert len(kws) <= 10

    def test_stop_words_excluded(self):
        kws = extract_keywords("the and or but python machine learning")
        stop = {"the", "and", "or", "but"}
        assert not stop.intersection(set(kws))

    def test_tech_terms_captured(self):
        kws = extract_keywords(
            "experience with pytorch tensorflow scikit-learn docker kubernetes"
        )
        joined = " ".join(kws)
        # At least some tech terms should appear
        assert any(t in joined for t in ["pytorch", "tensorflow", "docker"])


# ---------------------------------------------------------------------------
# Section extraction tests
# ---------------------------------------------------------------------------

class TestExtractSections:
    def test_always_has_full(self):
        sections = extract_sections("Some text here")
        assert "full" in sections

    def test_detects_skills_section(self):
        text = "Name: Jane\n\nSkills\nPython Docker AWS\n\nExperience\nWorked at Google"
        sections = extract_sections(text)
        assert "skills" in sections or "experience" in sections

    def test_returns_dict(self):
        assert isinstance(extract_sections("anything"), dict)


# ---------------------------------------------------------------------------
# MatchResult tests
# ---------------------------------------------------------------------------

class TestMatchResult:
    def _make_result(self, overall=75.0):
        return MatchResult(
            overall_score=overall,
            semantic_score=80.0,
            keyword_score=65.0,
            matched_keywords=["python", "pytorch", "docker"],
            missing_keywords=["kubernetes"],
            section_scores={"skills": 85.0, "experience": 72.0},
            recommendations=["Add missing keywords"],
        )

    def test_to_dict(self):
        d = self._make_result().to_dict()
        assert isinstance(d, dict)
        assert "overall_score" in d
        assert "recommendations" in d

    def test_summary_contains_score(self):
        s = self._make_result(75.0).summary()
        assert "75.0" in s

    def test_summary_grade_excellent(self):
        assert "Excellent" in self._make_result(85.0).summary()

    def test_summary_grade_poor(self):
        assert "Poor" in self._make_result(20.0).summary()


# ---------------------------------------------------------------------------
# ResumeMatcher integration tests
# ---------------------------------------------------------------------------

class TestResumeMatcher:
    @pytest.fixture(scope="class")
    def matcher(self):
        return ResumeMatcher()

    def test_match_returns_result(self, matcher):
        result = matcher.match(RESUME_STRONG, JD_ML)
        assert isinstance(result, MatchResult)

    def test_strong_resume_scores_higher(self, matcher):
        strong = matcher.match(RESUME_STRONG, JD_ML)
        weak   = matcher.match(RESUME_WEAK,   JD_ML)
        assert strong.overall_score > weak.overall_score, (
            f"Strong ({strong.overall_score}) should beat weak ({weak.overall_score})"
        )

    def test_score_in_range(self, matcher):
        result = matcher.match(RESUME_STRONG, JD_ML)
        assert 0 <= result.overall_score <= 100
        assert 0 <= result.semantic_score <= 100
        assert 0 <= result.keyword_score  <= 100

    def test_matched_keywords_are_in_both(self, matcher):
        result = matcher.match(RESUME_STRONG, JD_ML)
        resume_text = RESUME_STRONG.lower()
        jd_text     = JD_ML.lower()
        for kw in result.matched_keywords:
            # Allow partial match for bigrams
            assert any(part in resume_text for part in kw.split()), f"{kw} not in resume"
            assert any(part in jd_text     for part in kw.split()), f"{kw} not in JD"

    def test_recommendations_not_empty(self, matcher):
        result = matcher.match(RESUME_STRONG, JD_ML)
        assert len(result.recommendations) > 0

    def test_rank_returns_sorted(self, matcher):
        resumes = [
            ("Strong", RESUME_STRONG),
            ("Weak",   RESUME_WEAK),
        ]
        ranked = matcher.rank(resumes, JD_ML)
        scores = [r.overall_score for _, r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_returns_all(self, matcher):
        resumes = [("A", RESUME_STRONG), ("B", RESUME_WEAK)]
        ranked  = matcher.rank(resumes, JD_ML)
        assert len(ranked) == 2

    def test_identical_texts_score_high(self, matcher):
        result = matcher.match(JD_ML, JD_ML)
        assert result.overall_score >= 70, "Identical texts should score high"

    def test_empty_sections_handled(self, matcher):
        result = matcher.match("Python developer", "Looking for Python developer")
        assert isinstance(result, MatchResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
