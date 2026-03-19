"""
Resume / Job Description Matcher
Run: python3 app.py
"""

import re
import gradio as gr
import urllib.request
from src.matcher import ResumeMatcher

matcher = ResumeMatcher()


def extract_pdf_text(pdf_path):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        return "ERROR: Install PyMuPDF — run: pip3 install pymupdf"
    except Exception as e:
        return f"ERROR reading PDF: {str(e)}"


def scrape_job_url(url):
    if not url or not url.startswith("http"):
        return ""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
        html = re.sub(r"<[^>]+>", " ", html)
        html = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", html)
        html = re.sub(r"\s+", " ", html).strip()
        return html[:4000]
    except Exception as e:
        return f"ERROR fetching URL: {str(e)}"


def compute_ats_score(resume_text, jd_text):
    from src.matcher import extract_keywords, preprocess
    resume_lower = preprocess(resume_text)
    jd_keywords  = extract_keywords(jd_text, top_n=30)
    matched = [kw for kw in jd_keywords if kw in resume_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_lower]
    score = round((len(matched) / len(jd_keywords)) * 100, 1) if jd_keywords else 0
    return score, matched, missing


def run_match(pdf_file, resume_text, job_url, jd_text):
    if pdf_file is not None:
        resume = extract_pdf_text(pdf_file)
        if resume.startswith("ERROR"):
            return resume, "", "", ""
    elif resume_text.strip():
        resume = resume_text.strip()
    else:
        return "Please upload a PDF resume or paste your resume text.", "", "", ""

    if job_url.strip():
        jd = scrape_job_url(job_url.strip())
        if jd.startswith("ERROR"):
            return "", jd, "", ""
    elif jd_text.strip():
        jd = jd_text.strip()
    else:
        return "", "Please paste a job URL or paste the job description text.", "", ""

    result = matcher.match(resume, jd)
    ats_score, ats_matched, ats_missing = compute_ats_score(resume, jd)

    grade = (
        "Excellent" if result.overall_score >= 80 else
        "Good"      if result.overall_score >= 60 else
        "Fair"      if result.overall_score >= 40 else
        "Poor"
    )
    ats_grade = (
        "Will pass ATS"    if ats_score >= 70 else
        "May get filtered" if ats_score >= 40 else
        "Likely filtered by ATS"
    )

    score_md = f"""
## Match: {grade} — {result.overall_score:.1f} / 100

| Metric | Score | What it means |
|---|---|---|
| Semantic similarity | {result.semantic_score:.1f} / 100 | How well your experience matches |
| Keyword overlap | {result.keyword_score:.1f} / 100 | How many JD terms appear in resume |
| ATS score | {ats_score} / 100 | {ats_grade} |
"""

    matched_str = "  ".join([f"`{k}`" for k in result.matched_keywords]) or "None found"
    missing_str = "  ".join([f"`{k}`" for k in result.missing_keywords]) or "None missing!"
    keyword_md = f"""
### Matched keywords
{matched_str}

### Missing from your resume
{missing_str}
"""

    if result.section_scores:
        rows = "\n".join(
            f"| {sec.capitalize()} | {score:.1f}/100 |"
            for sec, score in sorted(result.section_scores.items(), key=lambda x: x[1], reverse=True)
        )
        section_md = f"| Section | Score |\n|---|---|\n{rows}"
    else:
        section_md = "Add clear section headers to your resume (Skills, Experience, Education, Projects)"

    recs_md = "\n\n".join(f"**{i}.** {r}" for i, r in enumerate(result.recommendations, 1))

    return score_md, keyword_md, section_md, recs_md


def fetch_jd_from_url(url):
    if not url.strip():
        return ""
    return scrape_job_url(url.strip())


with gr.Blocks(title="Resume Matcher") as demo:

    gr.Markdown("# Resume / Job Description Matcher\nML-powered ATS + Semantic matching")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Your Resume")
            pdf_input    = gr.File(label="Upload PDF resume", file_types=[".pdf"])
            resume_input = gr.Textbox(placeholder="Or paste your resume text here...", lines=12)

        with gr.Column():
            gr.Markdown("### Job Description")
            url_input  = gr.Textbox(placeholder="Paste LinkedIn / Naukri job URL here...", label="Job URL")
            fetch_btn  = gr.Button("Fetch JD from URL", variant="secondary")
            jd_input   = gr.Textbox(placeholder="Or paste job description here...", lines=12)

    match_btn = gr.Button("Analyse Match", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## Results")

    with gr.Row():
        with gr.Column():
            score_out   = gr.Markdown()
            keyword_out = gr.Markdown()
        with gr.Column():
            section_out = gr.Markdown()
            recs_out    = gr.Markdown()

    fetch_btn.click(fn=fetch_jd_from_url, inputs=[url_input], outputs=[jd_input])
    match_btn.click(
        fn=run_match,
        inputs=[pdf_input, resume_input, url_input, jd_input],
        outputs=[score_out, keyword_out, section_out, recs_out],
    )


if __name__ == "__main__":
    demo.launch()
