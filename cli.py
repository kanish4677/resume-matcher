"""
Resume Matcher — CLI
Usage examples:

    # Match a single resume against a JD
    python cli.py match --resume resume.txt --jd job.txt

    # Rank multiple resumes against one JD (recruiter mode)
    python cli.py rank --resumes r1.txt r2.txt r3.txt --jd job.txt

    # Run on built-in demo data
    python cli.py demo
"""

import argparse
import sys
from pathlib import Path
from src.matcher import ResumeMatcher


def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found — {path}")
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def cmd_match(args):
    resume = read_file(args.resume)
    jd     = read_file(args.jd)
    matcher = ResumeMatcher()
    result  = matcher.match(resume, jd)

    print("\n" + "="*60)
    print(result.summary())
    print("Section scores:")
    for sec, score in sorted(result.section_scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"  {sec:<16} {bar}  {score:.1f}")
    print("\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    print("="*60 + "\n")

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))


def cmd_rank(args):
    jd      = read_file(args.jd)
    resumes = [(Path(r).stem, read_file(r)) for r in args.resumes]
    matcher = ResumeMatcher()
    ranked  = matcher.rank(resumes, jd)

    print("\n" + "="*60)
    print(f"RANKING — {len(ranked)} resumes vs job description")
    print("="*60)
    for i, (name, result) in enumerate(ranked, 1):
        grade = (
            "Excellent" if result.overall_score >= 80 else
            "Good"      if result.overall_score >= 60 else
            "Fair"      if result.overall_score >= 40 else
            "Poor"
        )
        print(f"\n#{i}  {name}")
        print(f"    Score: {result.overall_score:.1f}/100  ({grade})")
        print(f"    Semantic: {result.semantic_score:.1f}  |  Keywords: {result.keyword_score:.1f}")
        print(f"    Matched: {', '.join(result.matched_keywords[:5])}")
    print("="*60 + "\n")


def cmd_demo(args):
    """Run a quick built-in demo without needing any files."""
    from app import SAMPLE_RESUME, SAMPLE_JD
    matcher = ResumeMatcher()

    print("\n[DEMO] Matching sample resume against sample ML Engineer JD...\n")
    result = matcher.match(SAMPLE_RESUME, SAMPLE_JD)

    print("="*60)
    print(result.summary())
    print("Section scores:")
    for sec, score in sorted(result.section_scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"  {sec:<16} {bar}  {score:.1f}")
    print("\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Resume ↔ Job Description Matcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # match
    p_match = sub.add_parser("match", help="Score one resume vs one JD")
    p_match.add_argument("--resume", required=True, help="Path to resume .txt file")
    p_match.add_argument("--jd",     required=True, help="Path to job description .txt file")
    p_match.add_argument("--json",   action="store_true", help="Also print full JSON output")
    p_match.set_defaults(func=cmd_match)

    # rank
    p_rank = sub.add_parser("rank", help="Rank multiple resumes vs one JD")
    p_rank.add_argument("--resumes", nargs="+", required=True, help="Paths to resume .txt files")
    p_rank.add_argument("--jd",      required=True,            help="Path to JD .txt file")
    p_rank.set_defaults(func=cmd_rank)

    # demo
    p_demo = sub.add_parser("demo", help="Run built-in demo data")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
