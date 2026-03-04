"""
RAG Pipeline - Stage 2: Embedding
Generates vector embeddings for documents using HuggingFace sentence-transformers.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings model for document encoding.

    Args:
        model_name: HuggingFace model identifier for sentence embeddings.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )
