"""
RAG Pipeline - Retrieval-Augmented Generation for GitHub Issue Debugging.

Pipeline stages:
1. Ingestion  - Load and parse GitHub issues from JSON
2. Embedding  - Generate vector embeddings
3. Vector Storage - Chunk and store in Chroma
4. Retrieval - Similarity search
5. Prompt Construction - Build LLM prompt from context
6. LLM Generation - Generate response with OpenAI
"""

from .ingestion import load_documents, merge_issue_files
from .embedding import get_embeddings
from .vector_storage import split_documents, create_vectorstore
from .retrieval import retrieve, docs_from_results
from .prompt_construction import build_prompt, build_context
from .llm_generation import get_llm, generate
from .pipeline import RAGPipeline

__all__ = [
    "load_documents",
    "merge_issue_files",
    "get_embeddings",
    "split_documents",
    "create_vectorstore",
    "retrieve",
    "docs_from_results",
    "build_prompt",
    "build_context",
    "get_llm",
    "generate",
    "RAGPipeline",
]
