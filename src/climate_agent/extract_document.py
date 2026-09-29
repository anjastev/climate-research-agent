import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_FILE = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "ipcc_ar6_summary_for_policymakers.pdf"
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "in", "is", "it", "of", "on", "or",
    "that", "the", "this", "to", "with",
}


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from every non-empty page of a PDF."""
    reader = PdfReader(str(pdf_path))
    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        # Remove blank lines and extra spaces.
        cleaned_text = "\n".join(
            line.strip()
            for line in page_text.splitlines()
            if line.strip()
        )

        if cleaned_text:
            extracted_pages.append((page_number, cleaned_text))

    return extracted_pages


def chunk_pdf_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[dict[str, int | str]]:
    """Split pages into overlapping word chunks while keeping page numbers."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks = []

    for page_number, page_text in pages:
        words = page_text.split()

        # Ignore pages containing only a short title or heading.
        if len(words) < 40:
            continue

        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]

            if len(chunk_words) >= 40:
                chunks.append(
                    {
                        "chunk_id": len(chunks) + 1,
                        "page": page_number,
                        "text": " ".join(chunk_words),
                    }
                )

            if end == len(words):
                break

            # Repeat some words in the next chunk to preserve context.
            start = end - overlap

    return chunks


def search_chunks(
    query: str,
    chunks: list[dict[str, int | str]],
    top_k: int = 5,
) -> list[dict[str, int | str]]:
    """Rank chunks by how often they contain the query keywords."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    query_words -= STOP_WORDS

    if not query_words:
        raise ValueError("The query has no searchable keywords.")

    results = []

    for chunk in chunks:
        text = str(chunk["text"]).lower()
        word_counts = Counter(re.findall(r"[a-zA-Z0-9]+", text))

        # Basic score: total occurrences of query words in this chunk.
        score = sum(word_counts[word] for word in query_words)

        if score > 0:
            result = dict(chunk)
            result["score"] = score
            results.append(result)

    # Highest score first; for ties, show the earlier page first.
    results.sort(
        key=lambda result: (
            -int(result["score"]),
            int(result["page"]),
        )
    )

    return results[:top_k]


def main() -> None:
    """Load the document once, then allow multiple searches."""
    if not PDF_FILE.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_FILE}")

    pages = extract_pdf_pages(PDF_FILE)

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    chunks = chunk_pdf_pages(pages)

    if not chunks:
        raise ValueError("No searchable text chunks were created.")

    total_words = sum(len(text.split()) for _, text in pages)

    print(f"PDF pages with text: {len(pages)}")
    print(f"Extracted words: {total_words}")
    print(f"Searchable chunks: {len(chunks)}")
    print("Enter a query to search. Type q to quit.")

    while True:
        query = input("\nSearch query: ").strip()

        if query.lower() in {"q", "quit", "exit"}:
            print("Search finished.")
            break

        if not query:
            print("Please enter a search query.")
            continue

        try:
            results = search_chunks(query, chunks)
        except ValueError as error:
            print(error)
            continue

        if not results:
            print("No matching passages found.")
            continue

        print(f"\nTop {len(results)} results:")

        for result in results:
            print(
                f"\nScore: {result['score']} | "
                f"Page: {result['page']} | "
                f"Chunk: {result['chunk_id']}"
            )
            print(f"{str(result['text'])[:500]}...")


if __name__ == "__main__":
    main()