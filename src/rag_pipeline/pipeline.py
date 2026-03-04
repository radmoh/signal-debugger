"""
RAG Pipeline - Orchestrator
Runs the full 6-stage pipeline for querying GitHub issues.
"""

from pathlib import Path
from typing import Optional

from langchain_core.vectorstores import VectorStore

from .ingestion import load_documents
from .vector_storage import split_documents, create_vectorstore
from .retrieval import retrieve, docs_from_results
from .prompt_construction import build_prompt
from .llm_generation import get_llm, generate


class RAGPipeline:
    """
    End-to-end RAG pipeline for GitHub issue debugging assistance.
    """

    def __init__(
        self,
        data_path: str = "SignalIssues.json",
        persist_directory: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.data_path = Path(data_path)
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._vectorstore: Optional[VectorStore] = None

    def index(self) -> "RAGPipeline":
        """
        Run stages 1-3: Ingest, embed, and store documents.
        Returns self for chaining.
        """
        documents = load_documents(str(self.data_path))
        split_docs = split_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._vectorstore = create_vectorstore(
            split_docs,
            persist_directory=self.persist_directory,
        )
        return self

    def query(
        self,
        query: str,
        k: int = 3,
        verbose: bool = False,
    ) -> str:
        """
        Run stages 4-6: Retrieve, build prompt, generate response.

        Args:
            query: User's crash report or question.
            k: Number of documents to retrieve.
            verbose: If True, print retrieved docs and scores.

        Returns:
            LLM-generated response.
        """
        if self._vectorstore is None:
            raise RuntimeError("Pipeline not indexed. Call index() first.")

        results = retrieve(self._vectorstore, query, k=k, with_scores=True)
        docs = docs_from_results(results)

        if verbose:
            for i, (doc, score) in enumerate(results, 1):
                print(f"Score: {score}")
                print(f"Issue: {doc.metadata.get('issue_number')}")
                print(f"Result #{i}:\n{doc.page_content[:500]}...\n")

        prompt = build_prompt(query, docs)
        llm = get_llm()
        return generate(prompt, llm)

    @property
    def vectorstore(self) -> Optional[VectorStore]:
        """Access the vector store for direct queries if needed."""
        return self._vectorstore
