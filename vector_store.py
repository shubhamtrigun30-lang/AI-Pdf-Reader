"""Page-aware text chunking and local FAISS embedding index helpers."""

from typing import Any

import numpy as np


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_pdf_pages(
    pages: list[str], source_name: str, chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split each page separately so each chunk keeps its original page number."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    step = chunk_size - overlap

    for page_number, page_text in enumerate(pages, start=1):
        cleaned_text = page_text.strip()
        if not cleaned_text:
            continue

        start = 0
        while start < len(cleaned_text):
            end = min(start + chunk_size, len(cleaned_text))
            text = cleaned_text[start:end].strip()
            if text:
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "text": text,
                        "page_number": page_number,
                        "source": source_name,
                    }
                )
            if end == len(cleaned_text):
                break
            start += step

    return chunks


def build_faiss_index(model: Any, chunks: list[dict[str, Any]]):
    """Embed chunks locally and return a FAISS index plus float32 vectors."""
    if not chunks:
        raise ValueError("There is no text to embed. Extract text from a PDF first.")

    # Import here so the Streamlit app can still show an install hint if FAISS is missing.
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "FAISS is not installed. In the active Python environment, run: "
            "python -m pip install faiss-cpu"
        ) from error

    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise RuntimeError("The embedding model returned an unexpected result.")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors


def retrieve_relevant_chunks(
    question: str, model: Any, index: Any, chunks: list[dict[str, Any]], top_k: int = 4
) -> list[dict[str, Any]]:
    """Embed a question and return its closest page-aware chunks from FAISS."""
    if not question or not question.strip():
        raise ValueError("Enter a question before searching the PDF.")
    if index is None or index.ntotal == 0 or not chunks:
        raise ValueError("There is no searchable PDF index. Extract the PDF and create embeddings first.")

    question_vector = model.encode(
        [question.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    question_vector = np.asarray(question_vector, dtype=np.float32)
    if question_vector.ndim != 2 or question_vector.shape[0] != 1:
        raise RuntimeError("The embedding model returned an unexpected question vector.")

    result_count = min(max(1, top_k), index.ntotal, len(chunks))
    scores, positions = index.search(question_vector, result_count)
    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0 or position >= len(chunks):
            continue
        result = dict(chunks[int(position)])
        result["similarity"] = float(score)
        results.append(result)

    if not results:
        raise RuntimeError("FAISS did not return any matching PDF chunks.")
    return results
