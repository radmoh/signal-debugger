"""
RAG Pipeline - Stage 4: Retrieval
Performs similarity search over the vector store to find relevant documents.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def retrieve(
    vectorstore: VectorStore,
    query: str,
    k: int = 3,
    with_scores: bool = True,
) -> list[Document] | list[tuple[Document, float]]:
    """
    Retrieve most relevant documents for a query.

    Args:
        vectorstore: Chroma VectorStore instance.
        query: User query string.
        k: Number of documents to retrieve.
        with_scores: If True, return (Document, score) tuples; else return Documents only.

    Returns:
        List of Documents or (Document, score) tuples.
    """
    if with_scores:
        return vectorstore.similarity_search_with_score(query, k=k)

    return vectorstore.similarity_search(query, k=k)


def docs_from_results(
    results: list[tuple[Document, float]],
) -> list[Document]:
    """
    Extract documents from similarity_search_with_score results.
    Use when passing to prompt construction without scores.

    Args:
        results: List of (Document, score) tuples.

    Returns:
        List of Document objects.
    """
    return [doc for doc, _ in results]
