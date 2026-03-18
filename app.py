"""
Resume / Job Description Matcher — Gradio Web App
Run: python app.py
"""

import json
import gradio as gr
from src.matcher import ResumeMatcher

matcher = ResumeMatcher()


# ---------------------------------------------------------------------------
# Sample data for demo
# ---------------------------------------------------------------------------

SAMPLE_RESUME = """
Jane Doe
jane.doe@email.com | github.com/janedoe | linkedin.com/in/janedoe

SUMMARY
Machine learning engineer with 2 years of experience building NLP and computer vision
systems. Skilled in Python, PyTorch, and deploying ML models to production using FastAPI
and Docker. Passionate about building scalable AI systems.

SKILLS
Languages: Python, SQL, JavaScript
ML/AI: PyTorch, TensorFlow, scikit-learn, Hugging Face Transformers, LangChain
Data: Pandas, NumPy, Spark, dbt
Tools: Docker, Kubernetes, Git, MLflow, Weights & Biases
Cloud: AWS (SageMaker, S3, EC2), GCP

EXPERIENCE
ML Engineer — TechCorp (2023–present)
- Built a document classification pipeline (BERT fine-tuned) processing 50k docs/day
- Reduced model inference latency by 40% using quantization and ONNX export
- Set up MLflow experiment tracking and automated retraining with Airflow
- Deployed REST API serving predictions with FastAPI + Docker on AWS ECS

Data Science Intern — StartupXYZ (2022)
- Built churn prediction model (XGBoost, 88% AUC) saving $200k/year in retention
- Created Tableau dashboards for product team KPIs
- Wrote SQL pipelines to clean and aggregate 10M+ rows of event data

PROJECTS
RAG Document Q&A System
- Built LangChain + ChromaDB pipeline for PDF question answering
- Integrated RAGAS evaluation framework; achieved faithfulness score of 0.87
- Deployed on Hugging Face Spaces

Crop Disease Detection
- Fine-tuned EfficientNet-B3 on PlantVillage dataset (54k images, 92.4% accuracy)
- Added GradCAM visual explanations for model interpretability
- Deployed as FastAPI service on Render

EDUCATION
B.Tech Computer Science — IIT Varanasi (2023)
CGPA: 8.6 / 10
"""

SAMPLE_JD = """
Senior ML Engineer — AI Product Team
DataDriven Inc. | Remote | Full-time

About the Role
We are looking for a skilled Machine Learning Engineer to join our growing AI team.
You will design, build, and deploy NLP and computer vision models that power our
core product features.

Responsibilities
- Design and train deep learning models for NLP tasks (classification, NER, summarization)
- Build scalable model serving infrastructure using FastAPI, Docker, and Kubernetes
- Implement MLOps best practices: experiment tracking, model versioning, CI/CD pipelines
- Collaborate with product and data engineering teams to ship ML features
- Monitor production models for data drift and performance degradation
- Mentor junior engineers on ML engineering best practices

Requirements
Must Have:
- 2+ years experience in machine learning or deep learning
- Strong Python skills and familiarity with PyTorch or TensorFlow
- Experience with NLP frameworks (Hugging Face Transformers, LangChain)
- Experience deploying ML models to production (REST APIs, Docker)
- Familiarity with cloud platforms (AWS, GCP, or Azure)
- Proficiency in SQL and data manipulation (Pandas, NumPy)

Nice to Have:
- Experience with MLflow, Weights & Biases, or similar experiment tracking
- Knowledge of vector databases (Pinecone, ChromaDB, Weaviate)
- Experience with Kubernetes or container orchestration
- Familiarity with LLMs, RAG pipelines, or prompt engineering
- Computer vision experience (YOLO, EfficientNet, object detection)

What We Offer
Competitive salary, remote-first culture, learning budget of $2000/year,
equity options.
"""


# ---------------------------------------------------------------------------
# Gradio functions
# ---------------------------------------------------------------------------

def run_match(resume_text: str, jd_text: str):
    if not resume_text.strip() or not jd_text.strip():
        return "⚠️ Please paste both a resume and a job description.", "", "", ""

    result = matcher.match(resume_text, jd_text)

    # Score card (markdown)
    grade = (
        "🟢 Excellent" if result.overall_score >= 80 else
        "🔵 Good"      if result.overall_score >= 60 else
        "🟡 Fair"      if result.overall_score >= 40 else
        "🔴 Poor"
    )
    score_md = f"""
## {grade} — {result.overall_score:.1f} / 100

| Metric | Score |
|---|---|
| 🧠 Semantic similarity | {result.semantic_score:.1f} / 100 |
| 🔑 Keyword overlap | {result.keyword_score:.1f} / 100 |
"""

    # Keywords
    matched_str = "  ".join([f"`{k}`" for k in result.matched_keywords]) or "_None found_"
    missing_str = "  ".join([f"`{k}`" for k in result.missing_keywords]) or "_None missing — great!_"
    keyword_md = f"""
### ✅ Matched keywords
{matched_str}

### ❌ Missing from your resume
{missing_str}
"""

    # Section scores
    if result.section_scores:
        rows = "\n".join(
            f"| {sec.capitalize()} | {score:.1f} / 100 |"
            for sec, score in sorted(result.section_scores.items(), key=lambda x: x[1], reverse=True)
        )
        section_md = f"| Section | Score |\n|---|---|\n{rows}"
    else:
        section_md = "_No matching sections detected — add clear section headers to your resume._"

    # Recommendations
    recs_md = "\n".join(f"- {r}" for r in result.recommendations)

    return score_md, keyword_md, section_md, recs_md


def load_samples():
    return SAMPLE_RESUME, SAMPLE_JD


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Resume ↔ JD Matcher", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 📄 Resume / Job Description Matcher
**Paste your resume and a job description to get an instant match score, keyword analysis, and tailored recommendations.**

Powered by sentence-transformers (`all-MiniLM-L6-v2`) + TF-IDF keyword analysis.
""")

    with gr.Row():
        with gr.Column():
            resume_input = gr.Textbox(
                label="Your Resume",
                placeholder="Paste your resume text here...",
                lines=20,
            )
        with gr.Column():
            jd_input = gr.Textbox(
                label="Job Description",
                placeholder="Paste the job description here...",
                lines=20,
            )

    with gr.Row():
        sample_btn = gr.Button("Load sample data", variant="secondary")
        match_btn  = gr.Button("Analyse match ▶", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## Results")

    with gr.Row():
        with gr.Column():
            score_out   = gr.Markdown(label="Match score")
            keyword_out = gr.Markdown(label="Keyword analysis")
        with gr.Column():
            section_out = gr.Markdown(label="Section breakdown")
            recs_out    = gr.Markdown(label="Recommendations")

    # Events
    match_btn.click(
        fn=run_match,
        inputs=[resume_input, jd_input],
        outputs=[score_out, keyword_out, section_out, recs_out],
    )
    sample_btn.click(
        fn=load_samples,
        inputs=[],
        outputs=[resume_input, jd_input],
    )

    gr.Markdown("""
---
### How scores are calculated
- **Semantic similarity (65%)** — encodes both texts as dense vectors using a transformer model and computes cosine similarity. Captures meaning beyond exact word matches.
- **Keyword overlap (35%)** — extracts top TF-IDF keywords from the job description and checks how many appear in your resume.
- **Section scores** — semantic similarity computed independently for each matched section (Skills, Experience, Education, etc.).
""")


if __name__ == "__main__":
    demo.launch(share=False)
