# Resume / Job Description Matcher 🎯

An ML-powered tool that scores how well a resume matches a job description. Uses **semantic embeddings** (sentence-transformers) + **TF-IDF keyword analysis** to give a weighted match score, keyword gap analysis, per-section scores, and actionable recommendations.

---

## Demo

![demo](assets/demo.gif)

---

## How it works

```
Resume text ──┐
              ├─► Preprocessing ─► Embedder (all-MiniLM-L6-v2) ─► Cosine similarity ─┐
JD text    ──┘                                                                         │
                                                                                       ├─► Weighted score (0–100)
Resume text ──┐                                                                        │
              ├─► TF-IDF keyword extraction ─► Set intersection ──────────────────────┘
JD text    ──┘
```

| Component | Weight | What it captures |
|---|---|---|
| Semantic similarity | 65% | Meaning, context, paraphrases |
| Keyword overlap | 35% | Exact / near-exact term matches |

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourname/resume-matcher
cd resume-matcher
pip install -r requirements.txt

# 2. Launch web app
python app.py
# Open http://localhost:7860

# 3. Or use the CLI
python cli.py demo                          # built-in demo
python cli.py match --resume r.txt --jd j.txt
python cli.py rank  --resumes r1.txt r2.txt --jd j.txt

# 4. Run tests
pytest tests/ -v
```

---

## Project structure

```
resume_matcher/
├── src/
│   └── matcher.py       # Core ML logic (matcher, embedder, preprocessor)
├── tests/
│   └── test_matcher.py  # Unit + integration tests (pytest)
├── app.py               # Gradio web UI
├── cli.py               # Command-line interface
├── requirements.txt
└── README.md
```

---

## Upgrading the embedder

The default embedder is `all-MiniLM-L6-v2` (80MB, very fast). To use a larger model:

```python
# src/matcher.py — change MODEL_NAME in STEmbedder
MODEL_NAME = "all-mpnet-base-v2"        # better quality, ~420MB
MODEL_NAME = "multi-qa-mpnet-base-cos-v1"  # optimised for Q&A / retrieval
```

The `TFIDFEmbedder` fallback activates automatically if `sentence-transformers` is not installed.

---

## Extending the project

| Idea | Difficulty | Impact |
|---|---|---|
| Add PDF parsing (PyMuPDF) | Easy | High — users can upload PDFs |
| Fine-tune embedder on resume/JD pairs | Hard | High — domain-specific scoring |
| Add named entity recognition for skills | Medium | Medium — structured skill extraction |
| Build a FastAPI backend | Medium | High — production deployment |
| Add database to store results | Medium | Medium — track improvement over time |
| Integrate with LinkedIn scraper | Hard | High — real JD data pipeline |

---

## Results on sample data

| Resume type | Semantic | Keywords | Overall |
|---|---|---|---|
| Strong ML engineer (matched domain) | ~85 | ~78 | **~82** |
| Weak (unrelated field) | ~18 | ~6  | **~14** |
| Partial match (related but junior) | ~62 | ~45 | **~56** |

---

## Tech stack

- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Similarity**: `scikit-learn` cosine_similarity
- **Keywords**: TF-IDF with bigrams
- **UI**: `gradio`
- **Tests**: `pytest`
- **Deploy**: Hugging Face Spaces (free)

---

## Deploying to Hugging Face Spaces

1. Create a new Space at huggingface.co/spaces (Gradio template)
2. Push this repo to the Space's git remote
3. Add a `README.md` with `sdk: gradio` in the YAML front matter
4. The Space will auto-build and deploy — free hosting!

---

## License

MIT
