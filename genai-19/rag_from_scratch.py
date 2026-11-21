"""
RAG from scratch (no framework) - Gen AI Masterclass Ep 19 (Hindi) - @DecodeWithVipin1306

Every snippet shown in the video, in one runnable file, in the order of the video:
  1. Problem in code: an LLM call with no context (knowledge cutoff, private docs)
  2. Problem in code: paste every document into the prompt
  3. Solution: ingestion  -> load, clean, chunk, embed, store (Postgres + pgvector)
  4. Solution: retrieve   -> search() with one SQL query (cosine distance)
  5. Solution: augment    -> build_prompt() with numbered sources
  6. Solution: generate   -> llm() and the 3-line rag()
  7. Keeping documents fresh: delete / re-ingest / mark superseded
  8. Four hidden details, each with its solution in code: search_v2() and rag_v2()

Setup (conda + Jupyter, as in the course):
  conda create -n rag python=3.11 -y && conda activate rag
  pip install openai "psycopg[binary]" pgvector numpy pypdf tiktoken
  Postgres 13+ with the pgvector extension installed. Easiest with Docker:
    docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
  export OPENAI_API_KEY="sk-..."      (Windows PowerShell: $env:OPENAI_API_KEY="sk-...")
  export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/postgres"

Run:
  python rag_from_scratch.py ingest ./school_docs          # PDFs / .txt / .md files
  python rag_from_scratch.py ask "Science practical: how many marks this year?"
  python rag_from_scratch.py ask-no-context "Science practical: how many marks this year?"

Models used: gpt-4o-mini (chat) and text-embedding-3-small (1,536-dim vectors).
Swap them for any chat / embedding model; if the embedding size changes, change vector(1536) too
and re-embed everything (one model per table).
"""

import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
import tiktoken
from openai import OpenAI
from pgvector.psycopg import register_vector
from psycopg.rows import namedtuple_row

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"   # returns 1,536 numbers per text
EMBED_DIM = 1536

client = OpenAI()                        # reads OPENAI_API_KEY from the environment


# =============================================================================
# 1. PROBLEM IN CODE: ask the LLM directly, with no context
# =============================================================================
def ask(question):
    """Rohan's first version: only the question goes to the model.
    The model answers from its training memory, which stopped at its
    knowledge cutoff and never saw the school's private circulars."""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": question}],   # only the question
    )
    return resp.choices[0].message.content

# ask("Science practical: how many marks this year?")
# -> "30 marks"      (old rule, said with full confidence)


# =============================================================================
# 2. PROBLEM IN CODE: paste every document into the prompt
# =============================================================================
def ask_with_everything(question, all_files):
    """Works for 3 small files, breaks for 2,000 pages (~12,00,000 tokens):
    the API returns context_length_exceeded, and even when it fits, every
    question pays for the whole library, is slow, and facts in the middle
    of a very long prompt get missed."""
    all_text = "\n".join(Path(f).read_text(encoding="utf-8") for f in all_files)
    return ask(all_text + "\n\nQuestion: " + question)
# -> openai.BadRequestError: ... 'code': 'context_length_exceeded'


# =============================================================================
# Database setup: Postgres + pgvector
# =============================================================================
db = None   # the Postgres connection, set by connect()


def connect():
    global db
    db = psycopg.connect(
        os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"),
        row_factory=namedtuple_row,      # rows behave like r.text, r.file, r.page
        autocommit=True,
    )
    db.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(db)                  # lets us pass numpy arrays as vectors
    db.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id         bigserial PRIMARY KEY,
            text       text,
            file       text,
            page       int,
            date       date,
            embedding  vector({EMBED_DIM}),
            superseded boolean DEFAULT false,   -- hidden detail 2
            audience   text    DEFAULT 'all'    -- hidden detail 4: 'all' | 'student' | 'teacher'
        )""")
    # Ep 18: an HNSW index keeps search fast once you have lakhs of chunks
    db.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
               "ON chunks USING hnsw (embedding vector_cosine_ops)")
    return db


# =============================================================================
# Helpers (simple on purpose; how to cut chunks well is the next episode)
# =============================================================================
_enc = None   # tokenizer, loaded on first use (downloads once, then cached)


def load_pages(path):
    """Yield (page_no, page_text). PDFs page by page, text files as page 1."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            yield i, page.extract_text() or ""   # scanned pages give "" (hidden detail 3)
    else:
        yield 1, path.read_text(encoding="utf-8")


def clean(text):
    """Drop page-number lines and repeated whitespace. Add your own header/footer rules."""
    lines = [ln for ln in text.splitlines() if not re.fullmatch(r"\s*(page\s*)?\d+\s*", ln, re.I)]
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()


def split(text, max_tokens=400):
    """Cut text into pieces of at most max_tokens tokens (equal-size chunks)."""
    global _enc
    _enc = _enc or tiktoken.get_encoding("cl100k_base")
    tokens = _enc.encode(text)
    for start in range(0, len(tokens), max_tokens):
        piece = _enc.decode(tokens[start:start + max_tokens]).strip()
        if piece:
            yield piece


def file_date(path):
    """Date of the document. Here: the file's modified date. In real apps, read it from the document."""
    return date.fromtimestamp(Path(path).stat().st_mtime)


# =============================================================================
# 3. SOLUTION: ingestion = load -> clean -> chunk -> embed -> store
# =============================================================================
def embed(text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding)          # 1,536 numbers


def ingest(path, audience="all"):
    """Run once per document, and again whenever a new document arrives."""
    n = 0
    for page_no, page_text in load_pages(path):           # load
        text = clean(page_text)                           # clean
        for piece in split(text, max_tokens=400):         # chunk
            db.execute(
                "INSERT INTO chunks (text, file, page, date, embedding, audience) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (piece, str(path), page_no, file_date(path), embed(piece), audience))  # embed + store
            n += 1
    print(f"ingested {n} chunks from {path}")


# =============================================================================
# 4. SOLUTION: retrieve = top-k chunks by cosine distance
# =============================================================================
def search(question, k=4):
    q = embed(question)                       # SAME embedding model as ingestion
    return db.execute("""
        SELECT text, file, date, page
        FROM chunks
        ORDER BY embedding <=> %s             -- <=> is cosine distance in pgvector
        LIMIT %s
    """, (q, k)).fetchall()


# =============================================================================
# 5. SOLUTION: augment = build the prompt (instructions, context, question)
# =============================================================================
SYSTEM = ("Answer ONLY from the context. "
          "If the answer is not in the context, say: Not found in documents. "
          "Cite sources like [1].")


def build_prompt(rows, question):
    context = "\n\n".join(
        f"[{i}] {r.file}, {r.date}, page {r.page}\n{r.text}"
        for i, r in enumerate(rows, start=1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
# 4 chunks x 400 tokens = ~1,600 tokens of context instead of ~12,00,000


# =============================================================================
# 6. SOLUTION: generate, and the whole RAG in 3 lines
# =============================================================================
def llm(messages):
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, messages=messages)   # temperature 0: no storytelling
    return resp.choices[0].message.content


def rag(question):
    rows = search(question)                  # retrieve
    messages = build_prompt(rows, question)      # augment
    return llm(messages)                         # generate
# rag("Science practical: how many marks this year?")
# -> "Science practical: 40 marks [1]"     (from the new circular, page 2)
# LangChain / LlamaIndex wrap exactly these three steps.


# =============================================================================
# 7. Keeping documents fresh (no retraining, the model is never touched)
# =============================================================================
def replace_document(old_file, new_file):
    db.execute("DELETE FROM chunks WHERE file = %s", (old_file,))   # remove old chunks
    ingest(new_file)                                            # add the new document


def supersede_document(old_file):
    """Keep the old chunks for history, but never retrieve them (hidden detail 2)."""
    db.execute("UPDATE chunks SET superseded = true WHERE file = %s", (old_file,))


# =============================================================================
# 8. FOUR HIDDEN DETAILS, each with its solution in code
#    1. search always returns something  -> similarity threshold + early return
#    2. old and new documents both "true" -> superseded flag in the WHERE clause
#    3. garbage in, garbage out           -> print and read the retrieved text
#    4. permission                        -> audience filter inside the search
# =============================================================================
def search_v2(question, audience, k=4, min_score=0.4):    # min_score: tune on your own questions
    q = embed(question)
    rows = db.execute("""
        SELECT text, file, date, page, 1 - (embedding <=> %s) AS score
        FROM chunks
        WHERE superseded = false                 -- 2: latest only
          AND audience IN ('all', %s)            -- 4: permission
        ORDER BY embedding <=> %s
        LIMIT %s
    """, (q, audience, q, k)).fetchall()
    return [r for r in rows if r.score >= min_score]         # 1: threshold


def rag_v2(question, audience="student", debug=False):
    rows = search_v2(question, audience)
    if not rows:                                             # 1: no LLM call at all
        return "Not found in documents. Please ask your teacher."
    if debug:                                                # 3: read what you retrieved
        for r in rows:
            print(f"{r.score:.2f}  {r.file} p.{r.page}  {r.text[:200]!r}")
    return llm(build_prompt(rows, question))


# =============================================================================
# Command line
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, arg = sys.argv[1], " ".join(sys.argv[2:])
    if cmd == "ask-no-context":
        print(ask(arg))
        sys.exit(0)
    connect()
    if cmd == "ingest":
        target = Path(arg)
        files = [p for p in target.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}] \
            if target.is_dir() else [target]
        for f in files:
            ingest(f)
    elif cmd == "ask":
        print(rag_v2(arg, audience="student", debug=True))
    else:
        print(__doc__)
