# RAG Pipeline – GitHub Issue Debugging Assistant

A Retrieval-Augmented Generation (RAG) pipeline that uses GitHub issues to help debug software problems. Given a crash report or question, it retrieves similar historical issues and generates root-cause analysis, past resolutions, and debugging steps.

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. **Ingestion** | `ingestion.py` | Load GitHub issues from JSON into LangChain Documents |
| 2. **Embedding** | `embedding.py` | Generate vector embeddings via HuggingFace |
| 3. **Vector Storage** | `vector_storage.py` | Chunk documents and store in Chroma |
| 4. **Retrieval** | `retrieval.py` | Similarity search for relevant issues |
| 5. **Prompt Construction** | `prompt_construction.py` | Build LLM prompt from context |
| 6. **LLM Generation** | `llm_generation.py` | Generate response with OpenAI |

## Project Structure

```
RAG/
├── src/
│   └── rag_pipeline/
│       ├── __init__.py
│       ├── ingestion.py          # Stage 1
│       ├── embedding.py          # Stage 2
│       ├── vector_storage.py     # Stage 3
│       ├── retrieval.py           # Stage 4
│       ├── prompt_construction.py # Stage 5
│       ├── llm_generation.py      # Stage 6
│       └── pipeline.py            # Orchestrator
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone and install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**

   Copy `.env.example` to `.env` and add your OpenAI API key:

   ```bash
   cp .env.example .env
   # Edit .env: OPENAI_API_KEY=your-key-here
   ```

3. **Prepare data**

   Place `SignalIssues.json` (GitHub issues JSON) in the project root, or merge paginated files:

   ```bash
   python main.py --merge issues_p1.json issues_p2.json issues_p3.json --data SignalIssues.json
   ```

## Usage

**CLI – single query**

```bash
python main.py "I am unable to load screenshots >25MB"
```

**CLI – interactive**

```bash
python main.py
# Enter your crash report when prompted
```

**Python API**

```python
from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(data_path="SignalIssues.json")
pipeline.index()

response = pipeline.query("I am unable to load screenshots >25MB", verbose=True)
print(response)
```

**Options**

- `--data PATH` – Path to issues JSON (default: `SignalIssues.json`)
- `--merge FILE [FILE ...]` – Merge issue files before indexing
- `-k N` – Number of documents to retrieve (default: 3)
- `--verbose` – Print retrieved documents and similarity scores
- `--persist DIR` – Persist vectorstore to directory for faster reuse

## Data Format

Expected JSON structure (GitHub API issues):

```json
[
  {
    "number": 12345,
    "title": "Issue title",
    "body": "Description...",
    "comments": "Comment text",
    "labels": [{"name": "bug"}],
    "created_at": "2025-01-01T00:00:00Z",
    "pull_request": null
  }
]
```

Pull requests are excluded during ingestion.

## License

MIT

<!-- Minor formatting tweak for test commit -->
