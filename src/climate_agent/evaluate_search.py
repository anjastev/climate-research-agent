import json
from pathlib import Path

from src.climate_agent.extract_document import (
    PDF_FILE,
    chunk_pdf_pages,
    extract_pdf_pages,
    search_chunks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_FILE = PROJECT_ROOT / "tests" / "data" / "search_eval.json"
TOP_K = 5


def main():
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    pages = extract_pdf_pages(PDF_FILE)
    chunks = chunk_pdf_pages(pages)

    page_hits = 0
    term_checks_passed = 0

    for question in questions:
        query = question["query"]
        expected_page = int(question["expected_page"])
        expected_terms = question.get("expected_terms", [])

        results = search_chunks(query, chunks, top_k=TOP_K)
        result_pages = [int(result["page"]) for result in results]

        page_found = expected_page in result_pages
        if page_found:
            page_hits += 1

        expected_page_text = " ".join(
            result["text"].lower()
            for result in results
            if int(result["page"]) == expected_page
        )
        missing_terms = [
            term for term in expected_terms
            if term.lower() not in expected_page_text
        ]

        terms_passed = not missing_terms
        if terms_passed:
            term_checks_passed += 1

        status = "PASS" if page_found and terms_passed else "FAIL"

        print(f"\n{status}: {query}")
        print(f"  Expected page: {expected_page}")
        print(f"  Pages in top {TOP_K}: {result_pages}")
        print(f"  Expected terms: {expected_terms or 'none'}")

        if missing_terms:
            print(f"  Terms not found on expected page: {missing_terms}")

    total = len(questions)
    print(f"\nPage Hit@{TOP_K}: {page_hits}/{total}")
    print(f"Term checks passed: {term_checks_passed}/{total}")


if __name__ == "__main__":
    main()