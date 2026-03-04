"""
RAG Pipeline - Stage 1: Ingestion
Loads GitHub issues from JSON and converts them to LangChain Documents.
"""

import json
from pathlib import Path

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document


def load_documents(data_path: str = "SignalIssues.json") -> list[Document]:
    """
    Load GitHub issues from JSON and convert to LangChain Documents.
    Filters out pull requests and structures content with metadata.

    Args:
        data_path: Path to JSON file containing GitHub issues.

    Returns:
        List of LangChain Document objects.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    loader = JSONLoader(
        file_path=str(data_path),
        jq_schema="""
.[] | 
{
    issue_number: .number,
    title: .title, 
    body: .body,
    comments: .comments,
    labels: .labels,
    created_at: .created_at,
    pull_request: .pull_request
}""",
        text_content=False,
    )
    raw_data = loader.load()

    documents: list[Document] = []
    for item in raw_data:
        content_dict = json.loads(item.page_content)
        if content_dict.get("pull_request") is not None:
            continue  # Skip pull requests

        label_names = [label["name"] for label in content_dict["labels"]] or ["none"]

        content = f"""
Issue Title: {content_dict['title']}

Description:
{content_dict['body']}

Comments:
{content_dict['comments']}

Labels:
{label_names}
"""

        doc = Document(
            page_content=content.strip(),
            metadata={
                "issue_number": content_dict["issue_number"],
                "created_at": content_dict["created_at"],
                "labels": label_names,
            },
        )
        documents.append(doc)

    return documents


def merge_issue_files(
    files: list[str],
    output_path: str = "SignalIssues.json",
) -> None:
    """
    Merge paginated issue JSON files into a single file.

    Args:
        files: List of JSON file paths to merge.
        output_path: Output path for merged file.
    """
    all_issues = []
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_issues.extend(json.load(f))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_issues, f, indent=2)
