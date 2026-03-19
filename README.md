# Resume / Job Description Matcher

An ML-powered tool that scores how well a resume matches a job description.

## Features
- Upload PDF resume or paste text
- Paste any job URL to auto-fetch the job description
- Semantic similarity score using sentence-transformers
- Keyword overlap score using TF-IDF
- ATS score — simulates what company filters actually do
- Per-section breakdown and actionable recommendations

## How to run locally

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the app
python3 app.py
```

Then open http://localhost:7860 in your browser.

## How it works

| Component | Weight | What it captures |
|---|---|---|
| Semantic similarity | 65% | Meaning and context using AI embeddings |
| Keyword overlap | 35% | Exact term matches like ATS systems |

## Tech stack
- sentence-transformers (all-MiniLM-L6-v2)
- scikit-learn (TF-IDF, cosine similarity)
- Gradio (web UI)
- PyMuPDF (PDF parsing)

## Project structure
```
resume-matcher/
├── src/
│   ├── __init__.py
│   └── matcher.py      # Core ML logic
├── tests/
│   └── test_matcher.py # Unit tests
├── app.py              # Web app
├── cli.py              # Command line interface
└── requirements.txt
```

## Run tests
```bash
pytest tests/ -v
```
