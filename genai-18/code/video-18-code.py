"""
Gen AI Masterclass · Episode 18 · Vector databases (ANN, HNSW, pgvector)
Every code snippet shown in the video, in one runnable Python file.
Channel: @DecodeWithVipin1306

Story: Rohan's doubt-solving app stores every student question as an embedding
vector. Brute-force search breaks at 10 lakh vectors; ANN indexes (IVF, HNSW)
and vector databases are the solution.

Install (Python 3.10+):
    pip install numpy faiss-cpu hnswlib "psycopg[binary]" pgvector elasticsearch qdrant-client

Run:
    python video-18-code.py

Sections 1-6 and 9 run on your laptop with no server.
Section 7 needs Postgres with the pgvector extension, section 8 needs Elasticsearch.
If a server is not running, that section prints "skipped" and the rest still runs.

Note: real embedding models give 1,536 numbers per vector and you may have 10 lakh
(1,000,000) vectors. To run in seconds on a laptop this file uses smaller,
made-up vectors (N and DIM below). Change them to see the real cost.
"""

import re
import time
import numpy as np

# ---------------------------------------------------------------------------
# 0. Data: fake "question embeddings" (normalized, grouped by topic)
# ---------------------------------------------------------------------------
N = 50_000          # saved questions   (video: 1,000,000)
DIM = 384           # numbers per vector (video: 1,536)
K = 3               # top-k results to show

rng = np.random.default_rng(42)
# Real embeddings live on a lower-dimensional "meaning" surface, so we make
# 32 hidden meaning-directions, mix them into DIM numbers, and add a little noise.
meaning = rng.normal(size=(N, 32)).astype("float32")
mixer = rng.normal(size=(32, DIM)).astype("float32")
vectors = meaning @ mixer + 0.5 * rng.normal(size=(N, DIM)).astype("float32")
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)      # length 1 (Episode 17)
ids = np.arange(N)
classes = rng.integers(6, 13, size=N)                            # class 6..12 metadata

query = vectors[123] + 0.1 * rng.normal(size=DIM).astype("float32")
query = (query / np.linalg.norm(query)).astype("float32")


# ---------------------------------------------------------------------------
# 1. PROBLEM: brute-force search with NumPy
#    Scores the query against EVERY saved vector, then sorts all of them.
# ---------------------------------------------------------------------------
def brute_force(q, k=K):
    scores = vectors @ q                 # one score per saved question (dot = cosine here)
    return np.argsort(-scores)[:k]       # sort all, keep top k

t = time.perf_counter()
top3 = brute_force(query)
print(f"[1] brute force top{K}: {top3.tolist()}  "
      f"({(time.perf_counter() - t) * 1000:.1f} ms, {N * DIM:,} multiplies)")
print(f"    at 10 lakh x 1,536 that is {1_000_000 * 1536:,} multiplies per query")


# ---------------------------------------------------------------------------
# 2. Developer thinking: a text cache does not help
#    In production this is redis.get(question_text); a dict shows the same idea.
# ---------------------------------------------------------------------------
cache = {"Photosynthesis क्या है": [12, 45, 78]}
for question in ["Photosynthesis क्या है", "photosynthesis समझाओ", "पौधे खाना कैसे बनाते हैं"]:
    hit = cache.get(question)            # redis.get(question) in a real app
    print(f"[2] cache {'HIT ' if hit else 'MISS'} for: {question}")
# Same meaning, different text -> different key -> cache miss.


# ---------------------------------------------------------------------------
# 3. ANN solution 1: IVF with FAISS (split vectors into groups, search a few)
# ---------------------------------------------------------------------------
try:
    import faiss

    nlist = 200                                   # video: 1,000 groups for 10 lakh vectors (~sqrt(N) to 4*sqrt(N))
    quantizer = faiss.IndexFlatIP(DIM)            # finds the nearest group centres
    ivf = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)                            # learn the group centres (k-means)
    ivf.add(vectors)                              # each vector joins its group
    ivf.nprobe = 10                               # search the 10 nearest groups
    D, I = ivf.search(query.reshape(1, -1), K)
    print(f"[3] FAISS IVF top{K}: {I[0].tolist()}")
except ImportError:
    ivf = None
    print("[3] skipped: pip install faiss-cpu")


# ---------------------------------------------------------------------------
# 4. Recall: how "approximate" is approximate?
#    recall@10 = how many of the true top 10 the ANN search also returned.
# ---------------------------------------------------------------------------
def recall_at_10(ann_search, test_queries):
    total = 0.0
    for q in test_queries:
        exact = set(brute_force(q, k=10).tolist())
        approx = set(ann_search(q, k=10))
        total += len(exact & approx) / 10          # e.g. 9 common ids -> 0.9
    return total / len(test_queries)

test_queries = vectors[rng.choice(N, 200, replace=False)] + 0.3 * rng.normal(size=(200, DIM)).astype("float32")
test_queries = (test_queries / np.linalg.norm(test_queries, axis=1, keepdims=True)).astype("float32")

if ivf is not None:
    for nprobe in (1, 10, 50):
        ivf.nprobe = nprobe
        r = recall_at_10(lambda q, k: ivf.search(q.reshape(1, -1), k)[1][0].tolist(), test_queries)
        print(f"[4] IVF nprobe={nprobe:<3} recall@10 = {r:.3f}")   # more groups -> more right, slower


# ---------------------------------------------------------------------------
# 5. ANN solution 2: HNSW with hnswlib (multi-floor graph)
#    M               = links per node (more = more accurate, more memory)
#    ef_construction = how carefully neighbours are picked while building
#    ef (ef_search)  = candidate list size at query time; change any time
# ---------------------------------------------------------------------------
try:
    import hnswlib

    hnsw = hnswlib.Index(space="cosine", dim=DIM)
    hnsw.init_index(max_elements=N, M=16, ef_construction=200)
    hnsw.add_items(vectors, ids)                  # build once
    for ef in (20, 50, 200):                      # ef must be >= k
        hnsw.set_ef(ef)                           # ef_search
        r = recall_at_10(lambda q, k: hnsw.knn_query(q, k=k)[0][0].tolist(), test_queries)
        t = time.perf_counter()
        labels, distances = hnsw.knn_query(query, k=K)
        ms = (time.perf_counter() - t) * 1000
        print(f"[5] HNSW ef={ef:<3} recall@10 = {r:.3f}  top{K}: {labels[0].tolist()}  ({ms:.2f} ms)")
    # Two lines of brute force (section 1) became one call: hnsw.knn_query(query, k=3)
except ImportError:
    print("[5] skipped: pip install hnswlib")


# ---------------------------------------------------------------------------
# 6. Memory maths (hidden detail 1) and today's quiz
# ---------------------------------------------------------------------------
def vector_ram_gb(n, dim, bytes_per_number=4):
    return n * dim * bytes_per_number / 1e9

print(f"[6] 10 lakh x 1,536 x 4 bytes = {vector_ram_gb(1_000_000, 1536):.2f} GB (+ graph links)")
print(f"    with halfvec (2 bytes)       = {vector_ram_gb(1_000_000, 1536, 2):.2f} GB")
# Quiz: vector_ram_gb(10_000_000, 768) -> try it yourself, then check the comments.


# ---------------------------------------------------------------------------
# 7. Vector database 1: Postgres + pgvector, from Python (psycopg)
#    Postgres already solves this with this solution: the pgvector extension.
#    Needs Postgres with pgvector installed. Set PG_DSN for your server.
# ---------------------------------------------------------------------------
import os
PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres@localhost:5432/postgres")

try:
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(PG_DSN, autocommit=True, connect_timeout=3) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)                         # lets psycopg send numpy arrays as vectors
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS questions")
        cur.execute(f"""
            CREATE TABLE questions (
                id              bigint PRIMARY KEY,
                text            text,
                class           int,
                embedding_model text,              -- hidden detail 4: remember which model
                embedding       vector({DIM})      -- video: vector(1536)
            )""")

        # Bulk load with COPY (fast); one row = id + metadata + vector
        with cur.copy("COPY questions (id, text, class, embedding_model, embedding) "
                      "FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int8", "text", "int4", "text", "vector"])
            for i in range(N):
                copy.write_row((int(i), f"question {i}", int(classes[i]), "demo-model", vectors[i]))

        # PROBLEM: a normal B-tree index cannot serve a similarity query
        cur.execute("CREATE INDEX questions_btree ON questions (embedding)")
        plan = cur.execute("EXPLAIN SELECT id FROM questions ORDER BY embedding <=> %s LIMIT 3",
                           (query,)).fetchall()
        print("[7] query plan with only a B-tree index:")
        for (line,) in plan:
            print("       ", re.sub(r"'\[[^\]]*\]'", "<query>", line))   # "Seq Scan" = every row checked
        cur.execute("DROP INDEX questions_btree")

        # SOLUTION: HNSW index with cosine distance
        t = time.perf_counter()
        cur.execute("CREATE INDEX ON questions USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)")
        print(f"[7] HNSW index built in {time.perf_counter() - t:.1f} s")

        cur.execute("SET hnsw.ef_search = 100")       # same knob as hnswlib set_ef
        rows = cur.execute("""
            SELECT id, text FROM questions
            WHERE class = %s
            ORDER BY embedding <=> %s
            LIMIT 3""", (10, query)).fetchall()
        print(f"[7] pgvector top3 for class 10: {rows}")

        # Hidden detail 2: filters + ANN. pgvector 0.8+ keeps scanning the index
        # until LIMIT rows pass the WHERE filter.
        try:
            cur.execute("SET hnsw.iterative_scan = relaxed_order")
            print("[7] iterative_scan on (pgvector 0.8+)")
        except psycopg.Error:
            print("[7] iterative_scan needs pgvector 0.8+ (your version is older)")

        # Hidden detail 1: halfvec stores 2 bytes per number (pgvector 0.7+)
        try:
            cur.execute(f"CREATE INDEX ON questions USING hnsw "
                        f"((embedding::halfvec({DIM})) halfvec_cosine_ops)")
            print("[7] halfvec index created (half the memory)")
        except psycopg.Error:
            print("[7] halfvec needs pgvector 0.7+ (your version is older)")
except ImportError:
    print("[7] skipped: pip install 'psycopg[binary]' pgvector")
except Exception as e:  # server not running, wrong password, etc.
    print(f"[7] skipped (no Postgres at PG_DSN): {type(e).__name__}")


# ---------------------------------------------------------------------------
# 8. Vector database 2: Elasticsearch kNN, from Python (elasticsearch-py 8.x)
#    num_candidates plays the same role as ef_search.
# ---------------------------------------------------------------------------
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
try:
    from elasticsearch import Elasticsearch, helpers

    es = Elasticsearch(ES_URL, request_timeout=3)
    es.info()
    es.options(ignore_status=404).indices.delete(index="questions")
    es.indices.create(index="questions", mappings={"properties": {
        "text": {"type": "text"},
        "class": {"type": "integer"},
        "embedding": {"type": "dense_vector", "dims": DIM, "similarity": "cosine"}}})
    helpers.bulk(es, ({"_index": "questions", "_id": int(i), "text": f"question {i}",
                       "class": int(classes[i]), "embedding": vectors[i].tolist()}
                      for i in range(5_000)))
    es.indices.refresh(index="questions")
    res = es.search(index="questions", knn={
        "field": "embedding", "query_vector": query.tolist(),
        "k": 3, "num_candidates": 100})
    print("[8] Elasticsearch top3:", [h["_id"] for h in res["hits"]["hits"]])
except ImportError:
    print("[8] skipped: pip install elasticsearch")
except Exception as e:
    print(f"[8] skipped (no Elasticsearch at ES_URL): {type(e).__name__}")


# ---------------------------------------------------------------------------
# 9. Vector database 3: Qdrant, from Python (qdrant-client)
#    ":memory:" runs Qdrant inside this script, no server needed.
#    For a real server: QdrantClient(url="http://localhost:6333")
# ---------------------------------------------------------------------------
try:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection(
        "questions",
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))
    client.upload_collection("questions", vectors=vectors[:5_000],
                             payload=[{"class": int(c)} for c in classes[:5_000]],
                             ids=list(range(5_000)))
    hits = client.query_points("questions", query=query, limit=3).points
    print("[9] Qdrant top3 (first 5,000 questions only):", [h.id for h in hits])
except ImportError:
    print("[9] skipped: pip install qdrant-client")


# ---------------------------------------------------------------------------
# 10. Hidden detail 3: recall drifts silently -> check it every day
# ---------------------------------------------------------------------------
def daily_recall_check(ann_search, threshold=0.95):
    r = recall_at_10(ann_search, test_queries)
    if r < threshold:
        print(f"[10] ALERT: recall dropped to {r:.3f}")   # send to Slack / email in production
    else:
        print(f"[10] recall OK: {r:.3f}")

try:
    hnsw.set_ef(200)
    daily_recall_check(lambda q, k: hnsw.knn_query(q, k=k)[0][0].tolist())
except NameError:
    pass
