import streamlit as st
from src.matcher import ResumeMatcher
import re
import urllib.request

st.set_page_config(page_title="Resume Matcher", page_icon="📄", layout="wide")

st.title("📄 Resume / Job Description Matcher")
st.markdown("**ML-powered ATS + Semantic matching — just like what companies use**")

@st.cache_resource
def load_matcher():
    return ResumeMatcher()

matcher = load_matcher()

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
        return f"Error fetching URL: {str(e)}"

def compute_ats_score(resume_text, jd_text):
    from src.matcher import extract_keywords, preprocess
    resume_lower = preprocess(resume_text)
    jd_keywords  = extract_keywords(jd_text, top_n=30)
    matched = [kw for kw in jd_keywords if kw in resume_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_lower]
    score = round((len(matched) / len(jd_keywords)) * 100, 1) if jd_keywords else 0
    return score, matched, missing

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    pdf_file = st.file_uploader("Upload PDF resume", type=["pdf"])
    resume_text = st.text_area("Or paste your resume text here", height=250)

with col2:
    st.subheader("Job Description")
    job_url = st.text_input("Paste LinkedIn / Naukri job URL")
    if st.button("Fetch JD from URL"):
        if job_url:
            with st.spinner("Fetching job description..."):
                fetched = scrape_job_url(job_url)
                st.session_state["jd_text"] = fetched
    jd_text = st.text_area("Or paste job description here", 
                            value=st.session_state.get("jd_text", ""), 
                            height=250)

st.markdown("---")

if st.button("Analyse Match ▶", type="primary", use_container_width=True):

    resume = ""
    if pdf_file is not None:
        try:
            import fitz
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            resume = "".join(page.get_text() for page in doc)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    elif resume_text.strip():
        resume = resume_text.strip()
    else:
        st.error("Please upload a PDF or paste your resume text.")
        st.stop()

    jd = jd_text.strip()
    if not jd:
        st.error("Please paste a job URL or job description text.")
        st.stop()

    with st.spinner("Analysing match..."):
        result  = matcher.match(resume, jd)
        ats_score, ats_matched, ats_missing = compute_ats_score(resume, jd)

    st.markdown("## Results")

    c1, c2, c3 = st.columns(3)
    grade = ("Excellent" if result.overall_score >= 80 else
             "Good"      if result.overall_score >= 60 else
             "Fair"      if result.overall_score >= 40 else "Poor")
    ats_grade = ("Will pass ATS"      if ats_score >= 70 else
                 "May get filtered"   if ats_score >= 40 else
                 "Likely filtered")

    c1.metric("Overall Match", f"{result.overall_score:.1f}/100", grade)
    c2.metric("Semantic Score", f"{result.semantic_score:.1f}/100")
    c3.metric("ATS Score", f"{ats_score}/100", ats_grade)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Matched keywords")
        st.write("  ".join([f"`{k}`" for k in result.matched_keywords]) or "None found")
        st.markdown("### Missing from your resume")
        st.write("  ".join([f"`{k}`" for k in result.missing_keywords]) or "None missing!")

    with col4:
        st.markdown("### Recommendations")
        for i, rec in enumerate(result.recommendations, 1):
            st.info(f"**{i}.** {rec}")

    if result.section_scores:
        st.markdown("### Section scores")
        for sec, score in sorted(result.section_scores.items(), key=lambda x: x[1], reverse=True):
            st.progress(int(score), text=f"{sec.capitalize()}: {score:.1f}/100")
