"""
Gen AI Masterclass (Hindi) - Episode 20: Chunking strategies
Every code snippet shown in the video, in one runnable file. Python 3.9+, no installs needed
(only numpy for the semantic-chunking demo:  pip install numpy).

Run:  python video-20-chunking-code.py
Each section prints a small demo. For every concept: PROBLEM first, then SOLUTION.
"""
import re
import ast
import numpy as np

# ---------------------------------------------------------------------------
# 1. FIXED-SIZE CHUNKING  (problem: cuts through sentences, numbers, tables)
# ---------------------------------------------------------------------------
print("=== 1. Fixed-size chunking ===")

def fixed_size_chunks_v1(text, size=500):
    # problem: blind scissors, cut every `size` characters
    return [text[i:i + size] for i in range(0, len(text), size)]

# a tiny circular whose boundary falls inside a sentence (size=40 just for the demo)
circular = "Board circular 12 Nov. Science practical के कुल अंक 40 होंगे, theory के 60 होंगे।"
for c in fixed_size_chunks_v1(circular, size=40):
    print("  chunk:", repr(c))          # chunk ends before '40 होंगे'... the number gets separated

# ---------------------------------------------------------------------------
# 2. OVERLAP  (solution: each chunk repeats the tail of the previous one)
# ---------------------------------------------------------------------------
print("\n=== 2. Overlap ===")

def fixed_size_chunks(text, size=500, overlap=100):
    step = size - overlap                      # 400
    return [text[i:i + size] for i in range(0, len(text), step)]

doc_10k = "x" * 10_000
print("  overlap=0   ->", len(fixed_size_chunks(doc_10k, overlap=0)), "chunks")    # 20
print("  overlap=100 ->", len(fixed_size_chunks(doc_10k, overlap=100)), "chunks")  # 25
print("  overlap=400 ->", len(fixed_size_chunks(doc_10k, overlap=400)), "chunks")  # 100 (too much)

# ---------------------------------------------------------------------------
# 3. SENTENCE-BASED CHUNKING
# ---------------------------------------------------------------------------
print("\n=== 3. Sentence-based chunking ===")

# problem: split(".") breaks decimals and ignores the Hindi purna viram '।'
print("  problem :", "Pass marks 3.5 हैं। Theory 60 है।".split("."))

# solution: split only after . ? ! । followed by a space
SENT_END = re.compile(r"(?<=[.?!।])\s+")

def sentence_chunks(text, max_chars=500):
    chunks, cur = [], ""
    for s in SENT_END.split(text):
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur.strip())
            cur = ""
        cur += s + " "
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

print("  solution:", sentence_chunks("Pass marks 3.5 हैं। Theory 60 है। Practical 40 है।", max_chars=40))
# For abbreviations like "Dr." use spaCy or NLTK sentence tokenizers.

# ---------------------------------------------------------------------------
# 4. STRUCTURE-BASED CHUNKING  (headings, code files, tables)
# ---------------------------------------------------------------------------
print("\n=== 4. Structure-based chunking ===")

FENCE = "`" * 3   # three backticks, written this way so this file stays readable

def split_markdown_buggy(md):
    # problem: a '#' comment inside a code block is treated as a heading
    sections, heading, buf = [], "Intro", []
    for line in md.splitlines() + ["# END"]:
        if line.startswith("#"):
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    return sections

def split_markdown(md):
    # solution: track code fences with an in_code flag
    sections, heading, buf, in_code = [], "Intro", [], False
    for line in md.splitlines() + ["# END"]:   # sentinel flushes the last section
        if line.startswith(FENCE):
            in_code = not in_code
        if line.startswith("#") and not in_code:
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    return sections

md = f"""# Science
## Practical
Practical marks: 40.
{FENCE}python
# marks jodo
total = 40 + 60
{FENCE}
## Theory
Theory marks: 60.
"""
print("  buggy   :", [h for h, _ in split_markdown_buggy(md)])   # '# marks jodo' shows up as a heading
print("  solution:", [h for h, _ in split_markdown(md)])         # only real headings

# code files: one function / class = one chunk (Python's ast module already solves this)
source = '''
def total(a, b):
    return a + b

class Marks:
    pass
'''
tree = ast.parse(source)
code_chunks = [ast.get_source_segment(source, node) for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
print("  code chunks:", len(code_chunks))

# tables: repeat the header on every row so "40" is never alone
table = [["Subject", "Practical", "Theory"], ["Science", 40, 60]]
header, *rows = table
table_chunks = [", ".join(f"{h}: {v}" for h, v in zip(header, row)) for row in rows]
print("  table chunk:", table_chunks[0])      # 'Subject: Science, Practical: 40, Theory: 60'

# ---------------------------------------------------------------------------
# 5. RECURSIVE SPLITTING  (headings -> sentences, only when a section is too big)
# ---------------------------------------------------------------------------
print("\n=== 5. Recursive splitting ===")

def recursive_chunks(md_text, max_chars=500):
    chunks = []
    for heading, body in split_markdown(md_text):
        pieces = [body] if len(body) <= max_chars else sentence_chunks(body, max_chars)
        for piece in pieces:
            chunks.append((heading, piece))
    return chunks

print("  chunks:", len(recursive_chunks(md)))
# LangChain wraps the same idea:
# RecursiveCharacterTextSplitter(
#     separators=["\n## ", "\n\n", "\n", "। ", ". ", " "],
#     chunk_size=500, chunk_overlap=50)

# ---------------------------------------------------------------------------
# 6. SEMANTIC CHUNKING  (cut where the meaning changes)
# ---------------------------------------------------------------------------
print("\n=== 6. Semantic chunking ===")

def semantic_chunks(sentences, embed, threshold=0.75, max_chars=1500):
    vecs = [embed(s) for s in sentences]            # unit-length vectors
    chunks, cur = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(vecs[i - 1], vecs[i]))   # cosine similarity
        too_big = len(" ".join(cur)) > max_chars    # solution: size guard
        if sim < threshold or too_big:              # problem: was only `sim < threshold`
            chunks.append(" ".join(cur))
            cur = []
        cur.append(sentences[i])
    chunks.append(" ".join(cur))
    return chunks

# Demo with a toy embedding (topic keyword -> direction). Replace with a real embedding model.
def toy_embed(s):
    topic = np.array([1.0, 0.0]) if "practical" in s.lower() else np.array([0.0, 1.0])
    return topic / np.linalg.norm(topic)

lecture = ["Practical needs a file.", "Practical has a viva.", "Exam date is 5 March.", "Exam is at 10 AM."]
print("  chunks:", semantic_chunks(lecture, toy_embed))

# ---------------------------------------------------------------------------
# 7. CHUNK SIZE: test it, don't guess
# ---------------------------------------------------------------------------
print("\n=== 7. Chunk size trade-off ===")

def search(question, chunks, k):
    # toy keyword search; replace with your vector search
    words = set(question.lower().split())
    return sorted(chunks, key=lambda c: -len(words & set(c.lower().split())))[:k]

def hit_rate(chunks, tests, k=4):
    hits = 0
    for question, must_contain in tests:            # 20-30 real questions
        top = search(question, chunks, k)           # your vector search
        hits += any(must_contain in c for c in top)
    return hits / len(tests)

docs = "Science practical marks are 40. Science theory marks are 60. Exam date is 5 March. " * 20
tests = [("science practical marks", "40"), ("exam date", "5 March")]
for size in (200, 400, 800, 1600):
    chunks = fixed_size_chunks(docs, size, overlap=size // 10)
    print(" ", size, hit_rate(chunks, tests))

# ---------------------------------------------------------------------------
# 8. METADATA  (a chunk is a record, not just text)
# ---------------------------------------------------------------------------
print("\n=== 8. Metadata ===")

# problem: only the bare text is stored
# db.add(text="Practical: 40 अंक", vector=embed("Practical: 40 अंक"))

# solution: a record with context + metadata
def make_record(doc, heading, piece, idx):
    header = f"{doc['title']} > {heading}"
    return {
        "id": f"{doc['id']}-{idx}",
        "embed_text": header + "\n" + piece,   # goes to the embedding model
        "text": piece,                         # goes into the prompt
        "metadata": {"file": doc["file"], "page": doc["page"],
                     "date": doc["date"], "role": doc["role"],
                     "heading": heading, "chunker": "v2"},
    }

doc = {"id": "circ-12nov", "title": "Board circular, 12 November", "file": "circular.pdf",
       "page": 2, "date": "2026-11-12", "role": "student"}
print(" ", make_record(doc, "Science > Practical", "Practical: 40 अंक", 0))

# ---------------------------------------------------------------------------
# 9. THREE HIDDEN DETAILS (one line each)
# ---------------------------------------------------------------------------
# size in tokens, not characters (Hindi uses several times more tokens):
#   len(chunk), len(tokenizer.encode(chunk))
# change chunking = re-chunk + re-embed everything, never mix old and new chunks:
#   db.delete(where={"chunker": "v1"})
# PDFs: join pages first, then chunk (keep page numbers in metadata):
#   text = "\n".join(clean(p) for p in pages)

# ---------------------------------------------------------------------------
# QUIZ (answers)
# ---------------------------------------------------------------------------
print("\n=== Quiz ===")
print("  size=500, overlap=100 ->", len(fixed_size_chunks(doc_10k, 500, 100)), "chunks")  # 25
print("  size=500, overlap=0   ->", len(fixed_size_chunks(doc_10k, 500, 0)), "chunks")    # 20
print("  size=500, overlap=400 ->", len(fixed_size_chunks(doc_10k, 500, 400)), "chunks")  # 100
