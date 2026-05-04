# Tiny RAG
A small, end-to-end retrieval-augmented generation (RAG) demo powered by [Ollama](https://ollama.com) and [Qdrant](https://qdrant.tech). Documents are embedded with Ollama, stored in Qdrant, and retrieved at query time to ground answers from a local chat model. Includes a CLI, a FastAPI server, a lightweight browser GUI with PDF / Markdown / text upload, and optional hybrid (dense + sparse BM42) retrieval.
This README doubles as a developer tutorial: it walks through the ingestion pipeline end-to-end (extract → chunk → embed → upsert), with pointers into `app.py` so you can read the implementation alongside the docs.
## What you'll learn
- How to extract text from PDFs (per-page), Markdown, and plain text.
- How to chunk documents into overlapping word windows that fit an embedding model's context.
- How to design a Qdrant payload that supports filtering, citation rendering, and idempotent re-ingest.
- How to expose ingest + retrieval over both an HTTP API and a CLI, sharing one pipeline.
- How to combine dense embeddings with a BM42 sparse vector and fuse them with Reciprocal Rank Fusion (RRF) for hybrid search.
- How to add a cross-encoder reranking stage on top of dense / hybrid retrieval to demo recall lift on ambiguous queries.
## Architecture
```
  user ──▶ GUI / curl ──▶ FastAPI (/chat, /ingest)
                          │
                          ├─ /ingest (multipart upload):
                          │    ├─ extract (pypdf for PDF, utf-8 for MD/TXT)
                          │    ├─ chunk   (word window with overlap, page-tracked)
                          │    ├─ embed   ──▶ Ollama (nomic-embed-text)        [dense]
                          │    │            └▶ fastembed BM42 (optional)      [sparse]
                          │    └─ upsert  ──▶ Qdrant (named vectors: dense [+ sparse])
                          │
                          └─ /chat (JSON):
                               ├─ embed query   ──▶ Ollama (+ BM42 if available)
                               ├─ vector search ──▶ Qdrant (ollama_demo_docs)
                               │                    ├─ hybrid: dense + sparse, fused with RRF
                               │                    └─ dense:  named-vector dense-only fallback
                               └─ generate      ──▶ Ollama (llama3.2)
```
The ingest pipeline is shared by three entrypoints — the `ingest` CLI (sample JSON), the `ingest-file` CLI (local PDF/MD/TXT), and the `POST /ingest` endpoint — all of which funnel into `ingest_chunks` in `app.py`. The /chat endpoint and `query` CLI share `search_documents`, which automatically picks hybrid or dense-only retrieval based on what the collection and runtime support.
## Project layout
- `app.py` — single-file CLI + FastAPI app. Subcommands: `ingest`, `ingest-file`, `query`, `traverse`, `serve`, `memory`, `quantize`, `bench`, `eval`.
- `dashboard.py` — Streamlit dashboard that drives `app.run_eval_sweep` from a browser and visualizes the resulting recall / MRR / latency / quantization memory metrics.
- `static/index.html` — single-file HTML/CSS/JS GUI served by `serve` (upload card + chat box + citations).
- `sample_data.json` — 8 example documents used by `ingest` (a JSON list of `{id, title, text, category}`).
- `qa_eval.json` — 8 labeled queries (`{query, expected_source}`) consumed by `bench` (recall@k + latency) and `eval` (recall@k + MRR).
- `requirements.txt` — Python deps: `qdrant-client`, `requests`, `fastapi`, `uvicorn`, `pydantic`, `pypdf`, `python-multipart`, plus optional `fastembed` (hybrid BM42 search) and `streamlit` + `pandas` (eval dashboard).
Key symbols inside `app.py`, if you want to read along:
- `extract_pdf_pages`, `extract_chunks_from_bytes` — file → tokens with optional page numbers.
- `chunk_words` — word-window chunker with overlap.
- `ensure_collection`, `_delete_chunks_by_source`, `ingest_chunks` — Qdrant collection + upsert plumbing (named dense vector, plus a sparse vector when fastembed is installed).
- `_sparse_available`, `_get_sparse_encoder`, `get_sparse_embedding` — lazy-loaded BM42 sparse encoder used by both ingest and query.
- `ingest_bytes` — top-level helper used by both the CLI and HTTP layer.
- `search_documents` — single retrieval entrypoint shared by `query` and `/chat`; returns `(points, search_mode)` where `search_mode` is `"hybrid"` or `"dense"`.
- `create_rag_app` — FastAPI factory mounting `/chat`, `/ingest`, `/health`, and the static GUI.
## Prerequisites
1. **Python 3.9+**
2. **Qdrant** running on `http://localhost:6333`
3. **Ollama** running on `http://localhost:11434`
4. An **embedding model** pulled in Ollama (default: `nomic-embed-text`, ~270 MB)
5. A **generation model** pulled in Ollama for `serve` (default: `llama3.2`, ~2 GB)
6. *(Optional)* **fastembed** installed in the venv to enable hybrid (dense + sparse BM42) search. It's listed in `requirements.txt`, so a default `pip install -r requirements.txt` already enables it.
### Start Qdrant
```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```
### Start Ollama and pull models
```bash
# install (macOS): brew install ollama
ollama serve &                # leave running in the background
ollama pull nomic-embed-text  # embeddings
ollama pull llama3.2          # chat model (only needed for `serve`)
```
## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> If you don't activate the venv, invoke its interpreter directly: `./.venv/bin/python app.py …`
## Quickstart tutorial (5 minutes)
This walks you from an empty Qdrant to a grounded chat response over your own document.
### Step 1 — Seed the collection with sample data
```bash
python app.py ingest
```
This drops + recreates the collection `ollama_demo_docs` with a named `dense` vector (cosine distance) and, if `fastembed` is installed, a `sparse` BM42 vector alongside it. It then ingests `sample_data.json` through the unified chunk pipeline. Output looks like:
```
Ingested 8 chunks across 8 sample documents into collection 'ollama_demo_docs' using model 'nomic-embed-text'.
```
### Step 2 — Add one of your own documents
```bash
python app.py ingest-file ./path/to/whitepaper.pdf --tags pricing,2024 --category docs
```
This extracts text (per page for PDFs), chunks it, embeds each chunk via Ollama, and upserts into the same collection. Existing chunks for `whitepaper.pdf` are replaced by default. Output:
```
  whitepaper.pdf (pdf, 12 pages): 38 chunks
Ingested 38 chunks total into collection 'ollama_demo_docs'.
```
### Step 3 — Inspect what was stored
```bash
python app.py traverse --limit 3
```
Scrolls through points and prints `id`, `category`, `title`, and a text preview — handy for sanity-checking payload shape.
### Step 4 — Ask a grounded question
```bash
python app.py query --query "What does the whitepaper say about pricing?" --limit 3
```
The CLI prints `Search mode: hybrid` (when fastembed + a sparse-enabled collection are both available) or `Search mode: dense` otherwise, followed by the top results.
Or start the server and use the chat endpoint:
```bash
python app.py serve
# then, in another shell:
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What does the whitepaper say about pricing?", "limit": 3}'
```
On startup `serve` also prints `Hybrid search: enabled` or `Hybrid search: disabled (install fastembed to enable).`. The response includes `answer`, a `search_mode` field (`"hybrid"` or `"dense"`), plus `citations[]` with `source`, `page_start`/`page_end`, `chunk_index`, and a text preview — enough to render "go to page N of file X" UI.
### Step 5 — Open the GUI
With `python app.py serve` running, open <http://127.0.0.1:8000>. The page has an **Add documents** card (file picker, category, tags, chunk size/overlap, replace toggle) and a chat box that surfaces citations inline.
## Ingestion deep dive
This section documents the full ingestion pipeline. The implementation lives in `app.py` and is intentionally one file so each step is easy to read in order.
### 1. Extraction (`extract_chunks_from_bytes`)
Dispatches on file extension:
- `.pdf` → `extract_pdf_pages` uses `pypdf` to return a list of `(page_number, text)` tuples (1-indexed). Pages with no extractable text become empty strings rather than errors. Scanned PDFs therefore yield zero chunks; OCR them first.
- `.md` / `.markdown` → decoded as UTF-8 (with `errors="replace"`), no per-page metadata.
- `.txt` → same as Markdown.
All three feed into a uniform token stream of `(word, page_or_none)` pairs. Adding a new format means adding one extractor and one branch in `extract_chunks_from_bytes`.
### 2. Chunking (`chunk_words`)
A simple word-window chunker:
- `chunk_size` words per window (default `400`).
- `chunk_overlap` words of overlap between consecutive windows (default `50`).
- Each chunk records `char_start` / `char_end` against the joined source text and `page_start` / `page_end` (the min/max page number of any token in the chunk).
- After the loop completes, every chunk gets `chunk_count` populated so consumers know the total without a second query.
Validations: `chunk_size > 0`, `0 ≤ chunk_overlap < chunk_size`. If you pass invalid values via the CLI/HTTP layer you'll get a clear error.
```
 source words: [w0 w1 w2 ... w399 w400 ... w749]
 chunk 0:      [w0 ............ w399]               size=400
 chunk 1:                  [w350 ............ w749] overlap=50, step=350
```
### 3. Embedding (`get_embedding` + `get_sparse_embedding`)
`get_embedding` calls Ollama's `/api/embed` (preferred) and falls back to `/api/embeddings` for older versions. Returns a `list[float]`. The dense vector size is discovered lazily from the first chunk on each ingest, so swapping embedding models "just works" for a fresh collection.
When `fastembed` is installed, `get_sparse_embedding` lazy-loads the `Qdrant/bm42-all-minilm-l6-v2-attentions` BM42 encoder (cached at module level) and returns a `models.SparseVector` per chunk. If fastembed is missing, sparse generation is silently skipped and the demo continues in dense-only mode.
Today dense embedding is sequential (one HTTP call per chunk). For larger corpora you can batch by sending an `input: [...]` list to `/api/embed`; the surface area is small enough that the change lives entirely in `ingest_chunks`.
### 4. Collection management (`ensure_collection`)
- If the collection doesn't exist, it's created with **named vectors**: a `dense` `VectorParams(size=<vector_size>, distance=Distance.COSINE)`. When `fastembed` is available, a `sparse` `SparseVectorParams()` config is added alongside it so Qdrant can store BM42 vectors per point.
- Keyword payload indexes are created for `category`, `source`, `source_type`, and `tags` so filtered search stays fast.
- If the collection already exists with the legacy unnamed-vector layout (older builds of this demo), `ensure_collection` raises a clear error telling you to drop and recreate it via `python app.py ingest`.
- If the collection exists and its `dense` vector size differs from what the current model produces, you get a clear error pointing at the model mismatch (instead of a confusing upsert failure later).
### 5. Upsert (`ingest_chunks`)
For each chunk:
- Embed it with Ollama (dense vector).
- If `fastembed` is installed, also produce a BM42 sparse vector via `get_sparse_embedding`.
- Build a payload (see schema below).
- Compute a deterministic point id: `uuid5(namespace, f"{source}#{chunk_index}")`. This means re-ingesting the same source overwrites the previous chunks one-for-one — the upsert is idempotent.
- Upsert the point with named vectors `{"dense": <vec>}`, plus `"sparse": <SparseVector>` when available.
If `replace_existing=True` (the default for `ingest-file` and `POST /ingest`), `_delete_chunks_by_source` first removes any prior chunks with the same `source` filter so renamed/shortened documents don't leave orphan chunks behind.
### 6. Re-use across entrypoints (`ingest_bytes`)
`ingest_bytes` is the single function the CLI (`ingest_files_command`) and HTTP layer (`POST /ingest`) both call. It returns an `IngestReport` with `source`, `title`, `source_type`, `chunks_ingested`, `pages`, and an optional `skipped_reason` — that's also what the HTTP response surfaces per-file. The sample-JSON path (`ingest_documents`) goes through `ingest_chunks` directly so its payload schema matches uploaded files exactly.
## Retrieval (`search_documents`)
`search_documents` is the single retrieval function shared by the `query` CLI and the `POST /chat` endpoint. It always embeds the query with Ollama for the dense side, then inspects the target collection to decide how to search. It returns `(points, search_mode)` where `search_mode` is `"hybrid"` or `"dense"`.
Mode selection:
- **hybrid** — chosen when the collection was created with both `dense` and `sparse` named vectors and `fastembed` is importable at query time. The function generates a BM42 sparse vector for the query and issues a single `client.query_points` call that prefetches from each index (`limit = max(limit * 4, 20)`) and fuses the results with `models.FusionQuery(fusion=models.Fusion.RRF)`. Reciprocal Rank Fusion lets sparse keyword matches and dense semantic matches each pull in their best candidates without one having to dominate.
- **dense** — used when the collection has no `sparse` config, when `fastembed` isn't installed, or when sparse encoding fails. The query runs against the named `dense` vector with `using="dense"`.
- **legacy fallback** — if the collection still has the old unnamed-vector schema, `search_documents` issues an unnamespaced dense query so existing data keeps working until you re-ingest.
A single payload `Filter` (built by `build_filter`) is applied uniformly across both prefetch branches in hybrid mode, so filter semantics don't change with the search mode.
## Reranking (optional third stage)
When `rerank=True` is passed to `search_documents`, the function adds a third stage on top of the dense or hybrid retrieval: it fetches a larger candidate pool (`max(limit*5, 25)`), runs each `(query, chunk_text)` pair through a fastembed cross-encoder, and re-orders by the reranker score before truncating to `limit`. The returned `search_mode` becomes `dense+rerank` or `hybrid+rerank` so downstream UIs can show what actually happened, and the per-point `score` field is replaced with the reranker score so the citation card surfaces the value that drove the new ordering.
The default model is `Xenova/ms-marco-MiniLM-L-6-v2` (~22 MB). Override per-call with `rerank_model=...` or globally with the `RERANK_MODEL` env var. Suggested alternatives surfaced through the same seam:
- `BAAI/bge-reranker-base` — stronger general-purpose reranker.
- `jinaai/jina-reranker-v1-tiny-en` — even smaller and faster than MiniLM.
- `jinaai/jina-reranker-v2-base-multilingual` — multilingual deployments.
The encoder is cached at module level keyed by model name, so swapping models live (CLI flag, /chat field, or GUI later on) doesn't pay the download/init cost twice for the same model.
If `fastembed` is not installed the rerank stage silently no-ops — the base mode result is returned unchanged — so demos work the same way regardless of optional dependencies.
### Three-mode demo (dense → hybrid → hybrid+rerank)
A quick SA-friendly script that runs the same query against three retrieval profiles. Best on a deliberately ambiguous query so the recall lift is visible:
```bash
Q="how do I keep vectors in memory after compression?"
# 1. Dense-only — force it by running against a collection without sparse,
#    or just inspect a baseline ordering.
python app.py query --query "$Q" --limit 3
# 2. Hybrid (auto-selected if the collection has sparse vectors).
python app.py query --query "$Q" --limit 3
# 3. Hybrid + cross-encoder rerank.
python app.py query --query "$Q" --limit 3 --rerank
```
The `Search mode:` line at the top of each result tells you which path actually ran. In the GUI, the same three modes are accessible by toggling the **Rerank with cross-encoder** checkbox (and dropping the sparse vectors from the collection if you want to demo dense-only).
## Chunk payload schema
Every point upserted into Qdrant carries the following payload. Fields beyond the basics are what enable filtering, page-jump UI, and re-ingest by source:
- `title` — display title (filename stem for uploads, JSON `title` for sample docs)
- `text` — the chunk text
- `category` — user-provided label
- `source` — stable identifier (filename for uploads, `sample:<id>` for sample docs)
- `source_type` — `pdf` / `markdown` / `text` / `sample`
- `chunk_index`, `chunk_count` — position of this chunk within its source (0-indexed)
- `char_start`, `char_end` — character range within the joined source text
- `page_start`, `page_end` — 1-indexed page range (PDF only)
- `page_count` — total pages in the source PDF
- `tags` — list of strings
- `created_at` — ISO 8601 UTC timestamp of ingestion
- `sample_id` — only present on points coming from `sample_data.json`
Point IDs are deterministic UUIDv5 hashes of `(source, chunk_index)`, so re-ingesting the same source with `replace=true` overwrites prior chunks idempotently.
## CLI reference
All commands accept the global flags `--qdrant-url`, `--ollama-url`, and `--model` (embedding model). Subcommands add their own.
### `ingest` — load `sample_data.json` (clean-slate)
```bash
python app.py ingest [--collection NAME] [--data-file PATH] \
                     [--chunk-size 400] [--chunk-overlap 50] \
                     [--quantization {none,scalar,binary,product}] \
                     [--no-always-ram] \
                     [--hnsw-m 16] [--hnsw-ef-construct 100]
```
Drops + recreates the collection, then ingests every document in the data file through the unified chunk pipeline. Override `--data-file` to point at any JSON list of `{id, title, text, category}` objects. `--quantization` (default `none`) is applied at collection-create time. `--always-ram` is on by default so the quantized vectors stay resident in RAM — pass `--no-always-ram` to allow them on disk. `--hnsw-m` and `--hnsw-ef-construct` configure the HNSW graph at create time (see [HNSW tuning](#hnsw-tuning-playground)).
### `ingest-file` — chunk + embed local PDF / MD / TXT files
```bash
python app.py ingest-file PATH [PATH ...] \
  [--collection NAME] [--category LABEL] [--tags a,b,c] \
  [--chunk-size 400] [--chunk-overlap 50] [--no-replace] \
  [--quantization {none,scalar,binary,product}] [--no-always-ram] \
  [--hnsw-m 16] [--hnsw-ef-construct 100]
```
- The collection is created lazily if it doesn't exist.
- Existing chunks for the same source filename are replaced by default; pass `--no-replace` to keep them.
- Supported extensions: `.pdf`, `.md`, `.markdown`, `.txt`.
- Per-file size limit: 25 MB (oversize files are skipped with a printed reason).
- Files that yield no extractable text (e.g. scanned PDFs without OCR) are skipped with a printed reason.
- `--quantization` is honored only when the collection is being created. If the collection already exists, the flag is ignored and a note is printed; use `python app.py quantize` to retrofit an existing collection in-place.
### `query` — semantic search
```bash
python app.py query --query "…" [--limit 3] [--collection NAME] \
                    [--rescore | --no-rescore] [--oversampling 2.0] \
                    [--rerank] [--rerank-model MODEL_ID] \
                    [--hnsw-ef 128]
```
Prints `Search mode: …` (one of `dense`, `hybrid`, `dense+rerank`, `hybrid+rerank`) followed by the top results with score, id, title, category, and a text preview.
`--rescore` / `--no-rescore` and `--oversampling` are passed through as Qdrant `QuantizationSearchParams` and only affect collections that have quantization configured. The recall demo: run the same query first with `--no-rescore` (fast, lower recall), then with `--rescore --oversampling 2.0` (slower, recall close to full precision).
`--rerank` enables the cross-encoder reranking stage described in [Reranking](#reranking-optional-third-stage). `--rerank-model` overrides the cross-encoder model id (defaults to `Xenova/ms-marco-MiniLM-L-6-v2`). When reranking is active the printed `score` is the cross-encoder score, not the dense / RRF score.
`--hnsw-ef` overrides Qdrant's per-query HNSW search width (`SearchParams(hnsw_ef=N)`). Higher = better recall, slower; pair it with `--hnsw-m` / `--hnsw-ef-construct` (build-time, on `ingest`) for a complete recall vs latency demo.
### `traverse` — scroll through stored points
```bash
python app.py traverse [--batch-size 4] [--limit 0] [--collection NAME]
```
Iterates through every point using Qdrant scroll pagination (no similarity required). Use `--limit N` to cap the number of points printed; `0` (default) means no limit.
### `serve` — run the FastAPI server + GUI
```bash
python app.py serve [--collection NAME] [--chat-model llama3.2] \
                    [--retrieval-limit 3] [--host 127.0.0.1] [--port 8000] \
                    [--rerank-default] [--rerank-model MODEL_ID]
```
Mounts `/`, `/health`, `/chat`, `/ingest`, and `/docs`. See the HTTP API section below.
`--rerank-default` makes `/chat` rerank by default; clients can still override per-request via `"rerank": false` in the JSON body. `--rerank-model` sets the default cross-encoder model id used when the request doesn't specify one.
### `memory` — collection stats + side-by-side quantization estimate
```bash
python app.py memory [--collection NAME]
```
Prints status, segment count, point count, dense vector size, distance, sparse-vector flag, and active quantization mode. Then renders an apples-to-apples RAM estimate of the dense vectors under each quantization mode (none / scalar / binary / product) with the active mode marked. Use this before and after `quantize` to demo the savings.
Example output:
```
Collection: ollama_demo_docs
  status:               green
  segments:             2
  points:               412
  dense vector size:    768
  distance:             Cosine
  sparse vectors:       yes
  active quantization:  scalar
Estimated dense-vector RAM by mode (lower bound, vectors only):
  mode             size   ratio vs full
  none          1.21 MB     1.0x
  scalar      309.00 KB     4.0x  <- active
  binary       38.62 KB    32.0x
  product      77.25 KB    16.0x
Active quantization saves 927.00 KB (75.0% vs full precision).
```
Estimates cover dense quantized vectors only; HNSW graph links and full-precision originals are stored separately.
### `quantize` — apply quantization to an existing collection in-place
```bash
python app.py quantize --mode {scalar|binary|product} [--no-always-ram] \
                       [--collection NAME]
```
Calls `client.update_collection(quantization_config=...)` so existing data is re-quantized in the background — no re-ingest required. Use `memory` a few seconds later to verify. To disable quantization entirely, drop the collection (e.g. `python app.py ingest`) and recreate without `--quantization`.
### `bench` — latency + recall@k against a labeled query set
```bash
python app.py bench [--collection NAME] [--queries-file qa_eval.json] \
                    [--limit 3] [--repeats 5] \
                    [--hnsw-ef 128] [--rerank] [--rerank-model MODEL_ID] \
                    [--rescore | --no-rescore] [--oversampling 2.0] \
                    [--compare-rerank]
```
Loads `qa_eval.json` (or any file you pass to `--queries-file`), runs one warmup per config followed by `--repeats` timed runs per query, and prints a one- or two-row table with end-to-end p50/p95/mean latency and recall@`limit`.
`qa_eval.json` is a JSON list of objects:
```json
[
  { "query": "How do I run Qdrant locally with Docker?", "expected_source": "sample:1" },
  { "query": "another question", "expected_sources": ["sample:2", "docs.pdf"] }
]
```
`expected_source` (string) and `expected_sources` (list[str]) are both honored — supplying either enables recall@k for that row. Latency is measured around `search_documents`, so it includes the Ollama embed call, the Qdrant query, and (when active) the cross-encoder rerank.
`--compare-rerank` is a convenience that runs the same bench twice in one CLI call — once with `rerank=False`, once with `rerank=True` — so the table shows both configs side by side. Pair it with `--hnsw-ef` to demo the recall lift of reranking on top of a tighter HNSW search.
Example output:
```
Bench: 8 queries x 5 repeats on 'ollama_demo_docs' (limit=3, hnsw_ef=128).
  config       mode             runs   p50 (ms)   p95 (ms)  mean (ms)        recall@3
  ----------------------------------------------------------------------------------
  no rerank    hybrid             40       42.3       58.1       45.7   87.5% (7/8)
  rerank       hybrid+rerank      40       72.4       91.7       76.1  100.0% (8/8)
```
### `eval` — recall@k + MRR sweep across retrieval configs
```bash
python app.py eval [--collection NAME] [--queries-file qa_eval.json] \
                   [--limit 3] [--repeats 3] [--hnsw-ef 128] \
                   [--rerank-model MODEL_ID] [--no-rerank] [--include-rescore] \
                   [--quantize-modes scalar,binary,product]
```
Where `bench` focuses on latency percentiles, `eval` focuses on retrieval *quality*. It sweeps a small grid of configs against the same query set and reports recall@k, mean reciprocal rank (MRR), and mean end-to-end latency per config.
Configs evaluated by default: `baseline` (the auto-selected dense / hybrid mode) and `rerank` (cross-encoder on top). `--no-rerank` skips the rerank config; `--include-rescore` adds `baseline+rescore` and (when reranking) `rerank+rescore` variants that pass `rescore=True` + `oversampling=2.0` through to Qdrant — only meaningful on quantized collections.
`--quantize-modes` accepts a comma-separated list from `{none, scalar, binary, product}` and runs the entire config grid once per requested mode. Each non-`none` mode is applied in-place via `client.update_collection` before its sweep, so you get one apples-to-apples table comparing retrieval quality across compression profiles — no re-ingest required. `none` is a *measurement-only* entry: it doesn't modify the collection, just adds a row tagged with whatever mode is currently active. Combine `none,scalar,binary,product` against a collection that starts in `none` to capture a true full-precision baseline alongside the quantized passes (the dashboard's **Precision tradeoff** tab uses that baseline to compute Δ columns). The output table gains a `quant` column. The collection is left in the **last** non-`none` requested mode; the harness prints a `Note: ...` line at the end with the exact `python app.py quantize --mode <orig>` command needed to restore the previous state. (Note: in-place updates can't switch a collection to `none`; reset by re-ingesting without `--quantization` if needed.)
**MRR (mean reciprocal rank)** is the average of `1/rank` for the first matching chunk in the top-k across all scored queries; misses contribute 0. Recall@k tells you whether the right chunk is *somewhere* in the top-k; MRR additionally rewards configs that rank it higher. Reranking typically lifts both, but MRR is what moves most when the right chunk was already in top-k but buried at position 3.
Eval entries support a richer schema than bench:
```json
[
  { "query": "How do I run Qdrant locally with Docker?", "expected_source": "sample:1" },
  { "query": "What does the pricing section say?", "expected_source": "whitepaper.pdf", "expected_page": 4 },
  { "query": "Multi-source", "expected_sources": ["a.md", "b.md"], "expected_pages": [2, 3] }
]
```
`expected_page` / `expected_pages` (1-indexed) tighten the match: a chunk only counts if its `page_start..page_end` range covers one of the expected pages. Useful for PDFs where a source-only match is too lenient (a long PDF can have many chunks; you want recall on the *right* chunk, not just any chunk).
Example output:
```
Eval: 8 queries x 3 repeats on 'ollama_demo_docs' (limit=3, hnsw_ef=128).
  config             quant    mode             runs  mean (ms)       recall@3     MRR
  --------------------------------------------------------------------------------------
  baseline           none     hybrid             24       46.2   87.5% (7/8)   0.792
  rerank             none     hybrid+rerank      24       77.8  100.0% (8/8)   0.958
```
With `--quantize-modes scalar,binary,product` you get one block of rows per quantization mode, all from a single CLI call:
```
Applying quantization='scalar' to 'ollama_demo_docs' in-place (always_ram=True)...
Applying quantization='binary' to 'ollama_demo_docs' in-place (always_ram=True)...
Applying quantization='product' to 'ollama_demo_docs' in-place (always_ram=True)...
Eval: 8 queries x 3 repeats on 'ollama_demo_docs' (limit=3, hnsw_ef=128).
  config             quant    mode             runs  mean (ms)       recall@3     MRR
  --------------------------------------------------------------------------------------
  baseline           scalar   hybrid             24       47.1   87.5% (7/8)   0.781
  rerank             scalar   hybrid+rerank      24       78.5  100.0% (8/8)   0.958
  baseline           binary   hybrid             24       45.0   62.5% (5/8)   0.583
  rerank             binary   hybrid+rerank      24       76.4   87.5% (7/8)   0.844
  baseline           product  hybrid             24       46.3   75.0% (6/8)   0.667
  rerank             product  hybrid+rerank      24       77.0   87.5% (7/8)   0.875

Note: collection started in quantization='none' but is now in 'product'. Run
'python app.py quantize --mode none' (or re-ingest without --quantization to
reset to none) to restore the previous state.
```
## Eval dashboard (Streamlit)
For an interactive view of the same data, the repo ships a Streamlit dashboard at `dashboard.py`. It calls `app.run_eval_sweep` directly so the metrics match the CLI table exactly — no separate code path or schema.
```bash
# One-time:
pip install -r requirements.txt    # pulls streamlit + pandas
# Run:
streamlit run dashboard.py
# Opens http://localhost:8501 by default.
```
The sidebar exposes every knob `app.py eval` accepts (collection, queries file, top-k, repeats, HNSW ef, rerank model, `--no-rerank`, `--include-rescore`, and the multi-select **Modes to sweep**). Click **Run eval** to execute the sweep against the live Qdrant + Ollama; the dashboard updates with:
- Top-line metric cards: best recall@k, best MRR, fastest config (mean latency), distinct quantization modes evaluated.
- **Quality** tab — grouped bar charts for recall@k and MRR per (config, quant).
- **Latency** tab — grouped bar chart selectable across `mean` / `p50` / `p95`.
- **Recall vs Latency** tab — scatter plot (top-left = ideal); each point is a (config, quant) combination.
- **Precision tradeoff** tab — single-pane comparison across quantization modes: compression ratio + RAM, average recall@k, MRR, and mean latency, plus Δ-vs-`none` columns when the baseline is in scope. Best read after running the sweep with `none` selected alongside `scalar / binary / product`.
- **Memory** tab — lower-bound dense-vector RAM estimate by mode for the live collection (matches the `app.py memory` table).
- **Raw data** tab — sortable DataFrame plus a **Download as JSON** button so you can share the run.
If you'd rather run the sweep from the CLI and view it later, dump it with `python app.py eval --output-json results.json --quantize-modes scalar,binary,product`, then paste the path into the dashboard's **Or load a saved JSON** field. Loaded JSON is the same `{meta, rows}` shape produced by `run_eval_sweep`, so notebooks and scripts can consume it directly too.
## HTTP API reference
### `GET /` — browser GUI
Serves `static/index.html`: an upload card (files, category, tags, chunk size/overlap, replace toggle) and a chat box that renders citations inline. ⌘/Ctrl+Enter sends from the textarea.
### `GET /health`
```json
{ "status": "ok" }
```
### `POST /chat`
Request body:
```json
{ "message": "How do I run Qdrant locally?", "limit": 3, "rerank": true }
```
`limit` is optional (defaults to `--retrieval-limit`, normally `3`; max `10`). `rerank` is optional too — when omitted the server-side default applies (controlled by `serve --rerank-default`). `rerank_model` (also optional) overrides the cross-encoder model id for that single request. Filter fields (`filter_category`, `filter_source`, `filter_source_type`, `filter_tags`) are also accepted and applied uniformly across the dense and sparse prefetch branches.
Response:
```json
{
  "answer": "…model output grounded in the retrieved chunks…",
  "collection": "ollama_demo_docs",
  "chat_model": "llama3.2",
  "search_mode": "hybrid",
  "citations": [
    {
      "id": "…uuid…",
      "score": 0.81,
      "title": "whitepaper",
      "category": "docs",
      "source": "whitepaper.pdf",
      "source_type": "pdf",
      "chunk_index": 7,
      "chunk_count": 38,
      "page_start": 4,
      "page_end": 5,
      "text_preview": "…first 180 chars…"
    }
  ]
}
```
`search_mode` is `"hybrid"` when the collection was created with sparse vectors and `fastembed` is installed; otherwise it is `"dense"`. In hybrid mode `score` is the RRF-fused score, not a cosine similarity.
### `POST /ingest`
Multipart form. Field reference:
- `files` (repeated, required) — one or more `.pdf`, `.md`, `.markdown`, or `.txt` files. 25 MB per file.
- `category` (string, default `"uploaded"`) — payload label written to every chunk.
- `tags` (string, default `""`) — comma-separated, parsed into a list and stored on every chunk.
- `replace` (bool, default `true`) — when true, prior chunks with the same `source` are deleted before upsert.
- `chunk_size` (int, default `400`, range `1..4000`) — words per chunk.
- `chunk_overlap` (int, default `50`, range `0..2000`) — words of overlap; must be `< chunk_size`.
Example:
```bash
curl -s http://127.0.0.1:8000/ingest \
  -F 'files=@whitepaper.pdf' \
  -F 'files=@notes.md' \
  -F 'category=docs' \
  -F 'tags=pricing,2024' \
  -F 'replace=true' \
  -F 'chunk_size=400' \
  -F 'chunk_overlap=50'
```
Response:
```json
{
  "collection": "ollama_demo_docs",
  "embed_model": "nomic-embed-text",
  "total_chunks": 42,
  "results": [
    {"source": "whitepaper.pdf", "title": "whitepaper", "source_type": "pdf", "pages": 12, "chunks_ingested": 38, "skipped_reason": null},
    {"source": "notes.md", "title": "notes", "source_type": "markdown", "pages": null, "chunks_ingested": 4, "skipped_reason": null}
  ]
}
```
Per-file errors (unsupported extension, oversize file, malformed PDF, no extractable text) appear in `results[].skipped_reason` and don't fail the whole request. Embedder/Qdrant failures bubble up as HTTP 500.
### `GET /docs`
FastAPI's auto-generated Swagger UI for `/chat` and `/ingest`.
## Configuration via environment variables
- `QDRANT_URL` — default `http://localhost:6333`
- `QDRANT_API_KEY` — API key for authenticated Qdrant instances (e.g. Qdrant Cloud). Unset by default (no auth).
- `OLLAMA_URL` — default `http://localhost:11434`
- `OLLAMA_MODEL` — embedding model, default `nomic-embed-text`
- `OLLAMA_CHAT_MODEL` — generation model used by `serve`, default `llama3.2`
- `RERANK_MODEL` — fastembed cross-encoder used when `--rerank` (CLI) or `rerank=true` (HTTP) is set; default `Xenova/ms-marco-MiniLM-L-6-v2`.
CLI flags always override env vars.
## Deploying with Qdrant Cloud
The app works against [Qdrant Cloud](https://cloud.qdrant.io) (or any authenticated Qdrant instance) with two environment variables. No code changes are needed.
### 1. Create a `.env` file
Copy the template below and fill in your cluster details:
```bash
# .env  —  never commit this file (it is already in .gitignore)
QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-api-key>
```
Find both values in the [Qdrant Cloud dashboard](https://cloud.qdrant.io) under your cluster's **Connection details** and **API keys** sections.
### 2. Migrate an existing local collection (optional)
If you already have data in a local Qdrant instance, use the [`qdrant/migration`](https://github.com/qdrant/migration) tool to copy it to the cloud. A ready-to-use wrapper script with validation, logging, and a `--dry-run` mode is available at [`examples/qdrant-to-qdrant/`](https://github.com/qdrant/migration/tree/main/examples/qdrant-to-qdrant) in that repo.
```bash
# First expose the gRPC port (6334) on your local Qdrant container if not already open:
docker stop qdrant && docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v /path/to/qdrant-storage:/qdrant/storage \
  qdrant/qdrant

# Dry run — validates config and checks connectivity, no data moved:
SOURCE_URL="http://host.docker.internal:6334" \
SOURCE_COLLECTION="ollama_demo_docs" \
TARGET_URL="https://<cluster-id>.<region>.cloud.qdrant.io:6334" \
TARGET_API_KEY="<your-api-key>" \
TARGET_COLLECTION="ollama_demo_docs" \
./examples/qdrant-to-qdrant/migrate.sh --dry-run

# Run the real migration:
# (same command without --dry-run)
```
> **macOS note:** Docker Desktop cannot reach `localhost` from inside a container via `--net=host`. Use `host.docker.internal` in place of `localhost` in `SOURCE_URL`.
### 3. Start the app
Source the `.env` file before launching:
```bash
set -a && source .env && set +a
./.venv/bin/python app.py serve --host 127.0.0.1 --port 8000
```
Or export inline for a one-off run:
```bash
QDRANT_URL="https://..." QDRANT_API_KEY="..." ./.venv/bin/python app.py serve
```
### 4. Verify
```bash
# Health check:
curl http://localhost:8000/health
# Confirm the cloud collection is visible:
curl http://localhost:8000/sources | python3 -m json.tool
```
### Troubleshooting cloud connections
- **`Unauthenticated` / `401`** — `QDRANT_API_KEY` is not set or is incorrect. Check the value in the Qdrant Cloud dashboard.
- **`connection refused` on port 6334** — gRPC port is not exposed by your local Docker container. Recreate it with `-p 6334:6334` (see step 2 above).
- **Migration fails with `ResourceExhausted`** — reduce `BATCH_SIZE` (e.g. `export BATCH_SIZE=16`) or add `--source.max-message-size` to the `docker run` call in `migrate.sh`.
## HNSW tuning playground
Qdrant's vector index is HNSW. Three knobs matter for the recall vs latency demo, and they all flow through this CLI:
- `--hnsw-m` (on `ingest` / `ingest-file`) — graph degree at build time. Qdrant default is 16. Higher = better recall and more RAM, slower index build. Applied only when the collection is being created; ignored on existing collections (drop and recreate to change).
- `--hnsw-ef-construct` (on `ingest` / `ingest-file`) — search width while building the graph. Default 100. Higher = better-quality graph, slower build.
- `--hnsw-ef` (on `query` / `bench`) — per-query search width. Higher = better recall, slower. This is the knob you usually tune live during a demo since it doesn't require a re-ingest.
The build-time params hit `models.HnswConfigDiff` on `create_collection`; the per-query param hits `models.SearchParams(hnsw_ef=...)` on `query_points`. Both are wired through `build_hnsw_config` and `build_search_params` in `app.py`, so the same plumbing serves the CLI, the chat endpoint, and the bench harness.
A quick playground script:
```bash
# Baseline: default HNSW (m=16, ef_construct=100), default per-query ef.
python app.py ingest
python app.py bench --repeats 5
# Wider build graph for better recall ceiling.
python app.py ingest --hnsw-m 32 --hnsw-ef-construct 200
python app.py bench --repeats 5 --hnsw-ef 128
# Show the recall lift of reranking on top of a tighter HNSW search.
python app.py bench --repeats 5 --hnsw-ef 128 --compare-rerank
```
## Quantization & memory reporting (SA demo flow)
A short walkthrough that lets a Solutions Architect tell a complete "quality, speed, cost" story end-to-end. It assumes Qdrant + Ollama are running and the venv is active.
```bash
# 1. Baseline: full-precision dense vectors.
python app.py ingest
python app.py memory
# 2. Apply scalar (int8) quantization in place — ~4x smaller, ~no recall loss.
python app.py quantize --mode scalar
python app.py memory   # active quantization should now read 'scalar'
# 3. Demo recall vs latency on a quantized collection.
python app.py query --query "hybrid search workflow" --no-rescore
python app.py query --query "hybrid search workflow" --rescore --oversampling 2.0
# 4. Or recreate from scratch with a different mode.
python app.py ingest --quantization binary       # 32x smaller (best on >=1024-dim models)
python app.py ingest --quantization product      # 16x smaller, lowest recall
```
Key points to surface during the demo:
- `scalar` (int8, quantile=0.99) is a near-free win on most embedding models.
- `binary` is the largest savings but only safe on ≥1024-dim embeddings; `nomic-embed-text` (768 dim) will lose noticeable recall.
- `product` (PQ x16) is a middle ground that preserves more structure than binary while still saving 16x.
- `--always-ram` (default) keeps quantized vectors hot in RAM. With original vectors on disk this gives RAM costs comparable to scalar/binary while preserving the ability to re-score from full precision.
- `--rescore` reads the original vectors to refine the top-k; `--oversampling N` fetches `N×limit` candidates first so re-scoring has more material to work with.
## Extending the demo
A few directions the code is structured to support:
- **Payload filtering.** The payload schema already includes `source`, `category`, `tags`, `source_type`, and `created_at`. Add `client.create_payload_index(...)` calls in `ensure_collection` and pass a `models.Filter` to `client.query_points` in `search_documents`.
- **Swap the sparse encoder.** Hybrid retrieval is wired through `_get_sparse_encoder` / `get_sparse_embedding`; point them at a different fastembed model (or a custom encoder that returns `(indices, values)`) to experiment with BM25, SPLADE, etc. Re-ingest after swapping so stored sparse vectors match the query encoder.
- **Tune the fusion.** `search_documents` uses `models.Fusion.RRF` with prefetch limit `max(limit * 4, 20)`. Try other fusion methods or weight the prefetches differently if you want one signal to dominate.
- **Batched embeddings.** Replace the per-chunk `get_embedding` loop in `ingest_chunks` with a batched call to Ollama's `/api/embed` (which accepts `input: [...]`). The rest of the pipeline doesn't need to change.
- **New file formats.** Add an extractor that returns a `List[(word, page_or_none)]` and one branch in `extract_chunks_from_bytes`.
## Troubleshooting
- **`zsh: command not found: python`** — use `python3` or the venv's interpreter (`./.venv/bin/python`).
- **`ModuleNotFoundError: No module named 'fastapi'` / `pypdf` / `multipart`** — the venv isn't active or deps aren't installed. Run `source .venv/bin/activate && pip install -r requirements.txt`, or call `./.venv/bin/python app.py …`.
- **Search mode is stuck on `dense` even though I want hybrid.** Either `fastembed` isn't installed in the active interpreter (re-run `pip install -r requirements.txt` and confirm with `python -c "import fastembed"`), or the collection was created before fastembed was installed and therefore has no `sparse` vector config. Drop the collection (e.g. `python app.py ingest`) so it's recreated with both `dense` and `sparse` configs and re-ingest your documents.
- **`Collection '…' was created with an older unnamed-vector schema`** — the collection predates the named-vector layout used for hybrid search. Drop and recreate it via `python app.py ingest`.
- **`/ingest` returns `No extractable text found`** — the PDF is likely scanned images. Run it through OCR (e.g. `ocrmypdf input.pdf output.pdf`) and re-upload.
- **`/ingest` returns a vector-size error** — the collection was created with a different embedding model. Drop the collection (`python app.py ingest` recreates it) or switch back to the matching `--model`.
- **`/chat` returns `Unable to generate embeddings from Ollama`** — Ollama isn't running, or the embedding model isn't pulled. Check with `curl http://localhost:11434/api/tags` and `ollama list`.
- **`/chat` returns `Unable to generate a response from Ollama`** — the chat model (default `llama3.2`) isn't pulled. Run `ollama pull llama3.2`.
- **First `/chat` or `/ingest` request is slow** — Ollama needs to load models into memory on cold start; subsequent requests are much faster.
- **`urllib3 NotOpenSSLWarning` about LibreSSL** — cosmetic; safe to ignore on macOS system Python.
