"""
RAG Pipeline - Stage 5: Prompt Construction
Builds the LLM prompt from retrieved context and user query.
"""

from langchain_core.documents import Document

SYSTEM_PROMPT = """You are a debugging assistant.
Use ONLY the provided Similar Historical Issues to answer.
If information is insufficient, say so explicitly.

Mention which issue numbers support your reasoning."""

USER_PROMPT_TEMPLATE = """Crash Report:
{query}

Similar Historical Issues:
{context_text}

Tasks:
1. Identify likely root cause.
2. Summarize past resolution.
3. Suggest debugging steps.
4. Provide confidence score (0-100%)."""


def build_context(documents: list[Document]) -> str:
    """
    Format retrieved documents into context string for the prompt.

    Args:
        documents: List of retrieved Documents.

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, doc in enumerate(documents, start=1):
        issue_num = doc.metadata.get("issue_number", "unknown")
        context_parts.append(f"\n[Issue {i} - #{issue_num}]\n{doc.page_content}")

    return "\n".join(context_parts) if context_parts else "No relevant issues found."


def build_prompt(query: str, documents: list[Document]) -> str:
    """
    Construct the full prompt for the LLM.

    Args:
        query: User's crash report or question.
        documents: Retrieved relevant documents.

    Returns:
        Complete prompt string.
    """
    context_text = build_context(documents)
    return f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(query=query, context_text=context_text)}"
