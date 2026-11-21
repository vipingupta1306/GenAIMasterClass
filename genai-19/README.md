# RAG from scratch: code + architecture (Gen AI Masterclass Ep 19, Hindi)

Companion files for the video "How Does AI Answer From Documents It Was Never Trained On? (RAG Explained From Scratch, With Code)" by @DecodeWithVipin1306.

| File | What it is |
|---|---|
| `rag_from_scratch.py` | Every code snippet from the video in one runnable Python file, in video order, with comments |
| `rag_from_scratch.ipynb` | The same code as a Jupyter notebook, one section per scene |
| `rag-architecture.png` / `.svg` | Architecture diagram: ingestion pipeline, Postgres + pgvector, query pipeline |

## Run it
```bash
conda create -n rag python=3.11 -y
conda activate rag
pip install openai "psycopg[binary]" pgvector numpy pypdf tiktoken

# Postgres with pgvector (Docker is the easiest way)
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16

export OPENAI_API_KEY="sk-..."          # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/postgres"

python rag_from_scratch.py ingest ./school_docs
python rag_from_scratch.py ask-no-context "Science practical: how many marks this year?"
python rag_from_scratch.py ask "Science practical: how many marks this year?"
```
Put your own PDFs, .txt or .md files in `school_docs/`. Or open the notebook in Jupyter and run the cells from top to bottom.

## What is inside
1. The problem in code: an LLM call with no context, and pasting every document into the prompt.
2. Ingestion: load, clean, chunk, embed, store.
3. Retrieve: `search()` with one SQL query (pgvector cosine distance).
4. Augment: `build_prompt()` with numbered sources.
5. Generate: `llm()` and the 3-line `rag()`.
6. Keeping documents fresh without retraining.
7. Four hidden details, each with its solution in code: `search_v2()` and `rag_v2()`.

Models: `gpt-4o-mini` and `text-embedding-3-small` (1,536 dims). You can swap them for any chat or embedding model. If the embedding size changes, change `vector(1536)` and re-embed everything.
