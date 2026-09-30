import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.climate_agent.extract_document import (
    PDF_FILE,
    chunk_pdf_pages,
    extract_pdf_pages,
)
from src.climate_agent.semantic_search import (
    MODEL_NAME,
    TOP_K,
    load_or_create_index,
    semantic_search,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_FILE = PROJECT_ROOT / "tests" / "data" / "semantic_search_eval.json"


def main():
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    print("Го вчитувам моделот...")
    model = SentenceTransformer(MODEL_NAME)

    print("Го читам PDF документот...")
    pages = extract_pdf_pages(PDF_FILE)
    chunks = chunk_pdf_pages(pages)
    embeddings = load_or_create_index(chunks, model)

    hits = 0

    for item in questions:
        results = semantic_search(
            query=item["query"],
            chunks=chunks,
            chunk_embeddings=embeddings,
            model=model,
            top_k=TOP_K,
        )

        expected_page = int(item["expected_page"])
        result_pages = [int(result["page"]) for result in results]
        passed = expected_page in result_pages

        if passed:
            hits += 1

        status = "PASS" if passed else "FAIL"
        print(f"\n{status}: {item['query']}")
        print(f"  Expected page: {expected_page}")
        print(f"  Pages in top {TOP_K}: {result_pages}")

    total = len(questions)
    print(f"\nSemantic Hit@{TOP_K}: {hits}/{total}")


if __name__ == "__main__":
    main()