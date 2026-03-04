"""
RAG Pipeline - Entry point for GitHub issue debugging assistant.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline, merge_issue_files

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline: Query GitHub issues for debugging assistance."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Crash report or question to search (interactive if omitted).",
    )
    parser.add_argument(
        "--data",
        default="SignalIssues.json",
        help="Path to issues JSON (default: SignalIssues.json)",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="FILE",
        help="Merge issue files before indexing (e.g., issues_p1.json issues_p2.json)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Number of documents to retrieve (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print retrieved documents and scores",
    )
    parser.add_argument(
        "--persist",
        default=None,
        help="Persist vectorstore to directory for reuse",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if args.merge:
        merge_issue_files(args.merge, str(data_path))
        print(f"Merged {len(args.merge)} files into {data_path}")

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        if not args.merge:
            print("Use --merge to create from issue files (e.g., issues_p1.json)")
        return

    pipeline = RAGPipeline(
        data_path=str(data_path),
        persist_directory=args.persist,
    )

    print("Indexing documents...")
    pipeline.index()
    print("Ready.\n")

    if args.query:
        query = args.query
    else:
        query = input("Enter your crash report or question: ").strip()
        if not query:
            print("No query provided.")
            return

    print("\nGenerating response...\n")
    response = pipeline.query(query, k=args.k, verbose=args.verbose)
    print(response)


if __name__ == "__main__":
    main()

