from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.climate_agent.extract_document import (
    PDF_FILE,
    chunk_pdf_pages,
    extract_pdf_pages,
)
from src.climate_agent.semantic_search import (
    MODEL_NAME as EMBEDDING_MODEL_NAME,
    TOP_K,
    load_or_create_index,
    semantic_search,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_MODEL_NAME = "gpt-4.1-mini"


def build_context(results):
    """Подготвува контекст со означени извори за LLM-от."""
    context_parts = []

    for source_number, result in enumerate(results, start=1):
        context_parts.append(
            f"[{source_number}] "
            f"PDF page {result['page']}, chunk {result['chunk_id']}\n"
            f"{result['text']}"
        )

    return "\n\n".join(context_parts)


def answer_question(question, results, client):
    """Генерира одговор врз основа на пронајдените пасуси."""
    context = build_context(results)

    response = client.responses.create(
        model=LLM_MODEL_NAME,
        instructions=(
            "You are a climate research assistant. "
            "Answer the user's question using only the provided source excerpts. "
            "If the excerpts do not contain enough information, say so clearly. "
            "Do not use outside knowledge or invent facts. "
            "Cite every factual claim with the source number, for example [1] or [2]. "
            "Answer in the same language as the user's question."
        ),
        input=(
            f"Question:\n{question}\n\n"
            f"Source excerpts:\n{context}"
        ),
    )

    return response.output_text


def main():
    if not PDF_FILE.exists():
        raise FileNotFoundError(f"PDF документот не е пронајден: {PDF_FILE}")

    try:
        client = OpenAI()
    except Exception as error:
        raise RuntimeError(
            "Не можам да го иницијализирам OpenAI клиентот. "
            "Провери дали OPENAI_API_KEY е поставен во овој терминал."
        ) from error

    print("Го вчитувам embedding моделот...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Го читам PDF документот...")
    pages = extract_pdf_pages(PDF_FILE)
    chunks = chunk_pdf_pages(pages)

    chunk_embeddings = load_or_create_index(chunks, embedding_model)

    print("\nClimate Research RAG е подготвен.")
    print("Внеси q за да завршиш.")

    while True:
        question = input("\nПрашање: ").strip()

        if question.lower() == "q":
            break

        if not question:
            print("Внеси прашање.")
            continue

        results = semantic_search(
            query=question,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            model=embedding_model,
            top_k=TOP_K,
        )

        if not results:
            print("Не најдов релевантни пасуси во документот.")
            continue

        try:
            answer = answer_question(question, results, client)
        except Exception as error:
            print(f"Грешка при повикот до LLM: {error}")
            continue

        print("\nОдговор:")
        print(answer)

        print("\nИзвори:")
        for source_number, result in enumerate(results, start=1):
            print(
                f"[{source_number}] PDF page {result['page']}, "
                f"chunk {result['chunk_id']} "
                f"(similarity: {result['score']:.3f})"
            )


if __name__ == "__main__":
    main()