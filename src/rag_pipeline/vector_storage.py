"""
RAG Pipeline - Stage 3: Vector Storage
Chunks documents and stores them in Chroma vector database.
"""

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embedding import get_embeddings


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split documents into smaller chunks for embedding and retrieval.

    Args:
        documents: List of LangChain Documents.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Overlap between chunks to preserve context.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vectorstore(
    documents: list[Document],
    persist_directory: Optional[str] = None,
    collection_name: str = "github_issues",
) -> VectorStore:
    """
    Create Chroma vector store from documents.

    Args:
        documents: Chunked documents to embed and store.
        persist_directory: Optional path to persist the database.
        collection_name: Chroma collection name.

    Returns:
        Chroma VectorStore instance.
    """
    embeddings = get_embeddings()

    if persist_directory:
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_path),
            collection_name=collection_name,
        )

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
    )
