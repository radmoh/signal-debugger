"""
RAG Pipeline - Stage 6: LLM Generation
Invokes the language model to generate responses from the constructed prompt.
"""

from langchain_openai import ChatOpenAI


def get_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> ChatOpenAI:
    """
    Initialize the ChatOpenAI LLM.

    Args:
        model: OpenAI model identifier.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        ChatOpenAI instance.
    """
    return ChatOpenAI(model=model, temperature=temperature)


def generate(prompt: str, llm: ChatOpenAI | None = None) -> str:
    """
    Generate LLM response for the given prompt.

    Args:
        prompt: Full prompt string.
        llm: Optional ChatOpenAI instance. Creates default if None.

    Returns:
        Generated response text.
    """
    if llm is None:
        llm = get_llm()

    response = llm.invoke(prompt)
    return response.content
