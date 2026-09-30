import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.climate_agent.extract_document import (
    PDF_FILE,
    chunk_pdf_pages,
    extract_pdf_pages,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
EMBEDDINGS_FILE = CACHE_DIR / "climate_embeddings.npy"
CHUNKS_FILE = CACHE_DIR / "climate_chunks.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def get_chunks_fingerprint(chunks):
    """Создава отпечаток за проверка дали chunks-овите се сменети."""
    content = "\n".join(chunk["text"] for chunk in chunks)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_or_create_index(chunks, model):
    """Ги вчитува зачуваните embeddings или ги пресметува и зачувува."""
    fingerprint = get_chunks_fingerprint(chunks)

    if EMBEDDINGS_FILE.exists() and CHUNKS_FILE.exists():
        with CHUNKS_FILE.open("r", encoding="utf-8") as file:
            cached_data = json.load(file)

        if (
            cached_data.get("model_name") == MODEL_NAME
            and cached_data.get("fingerprint") == fingerprint
        ):
            embeddings = np.load(EMBEDDINGS_FILE)
            print("Го користам постоечкиот локален индекс.")
            return embeddings

    print(f"Креирам embeddings за {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)

    with CHUNKS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "fingerprint": fingerprint,
                "chunk_count": len(chunks),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Индексот е зачуван во: {CACHE_DIR}")
    return embeddings


def semantic_search(query, chunks, chunk_embeddings, model, top_k=TOP_K):
    """Ги рангира chunks-овите според сличноста со прашањето."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    best_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in best_indices:
        chunk = chunks[index]
        results.append(
            {
                "score": float(scores[index]),
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
        )

    return results


def main():
    if not PDF_FILE.exists():
        raise FileNotFoundError(f"PDF документот не е пронајден: {PDF_FILE}")

    print("Го вчитувам моделот...")
    model = SentenceTransformer(MODEL_NAME)

    print("Го читам PDF документот...")
    pages = extract_pdf_pages(PDF_FILE)
    chunks = chunk_pdf_pages(pages)

    chunk_embeddings = load_or_create_index(chunks, model)

    print("\nСемантичкото пребарување е подготвено.")
    print("Внеси q за да завршиш.")

    while True:
        query = input("\nПрашање: ").strip()

        if query.lower() == "q":
            break

        if not query:
            print("Внеси прашање.")
            continue

        results = semantic_search(
            query=query,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            model=model,
        )

        print(f"\nТоп {len(results)} резултати:")

        for result in results:
            print(
                f"\nSimilarity: {result['score']:.3f}"
                f" | Page: {result['page']}"
                f" | Chunk: {result['chunk_id']}"
            )
            print(result["text"][:700])


if __name__ == "__main__":
    main()