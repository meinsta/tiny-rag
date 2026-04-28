# Tiny RAG
A small, end-to-end retrieval-augmented generation (RAG) demo powered by [Ollama](https://ollama.com) and [Qdrant](https://qdrant.tech). Documents are embedded with Ollama, stored in Qdrant, and retrieved at query time to ground answers from a local chat model. Includes a CLI, a FastAPI server, and a lightweight browser GUI with PDF / Markdown / text upload.
This README doubles as a developer tutorial: it walks through the ingestion pipeline end-to-end (extract → chunk → embed → upsert), with pointers into `app.py` so you can read the implementation alongside the docs.
## What you'll learn
- How to extract text from PDFs (per-page), Markdown, and plain text.
- How to chunk documents into overlapping word windows that fit an embedding model's context.
- How to design a Qdrant payload that supports filtering, citation rendering, and idempotent re-ingest.
- How to expose ingest + retrieval over both an HTTP API and a CLI, sharing one pipeline.
## Architecture
```
  user ──▶ GUI / curl ──▶ FastAPI (/chat, /ingest)
                          │
                          ├─ /ingest (multipart upload):
                          │    ├─ extract (pypdf for PDF, utf-8 for MD/TXT)
                          │    ├─ chunk   (word window with overlap, page-tracked)
                          │    ├─ embed   ──▶ Ollama (nomic-embed-text)
                          │    └─ upsert  ──▶ Qdrant (rich payload per chunk)
                          │
                          └─ /chat (JSON):
                               ├─ embed query   ──▶ Ollama
                               ├─ vector search ──▶ Qdrant (ollama_demo_docs)
                               └─ generate      ──▶ Ollama (llama3.2)
```
The ingest pipeline is shared by three entrypoints — the `ingest` CLI (sample JSON), the `ingest-file` CLI (local PDF/MD/TXT), and the `POST /ingest` endpoint — all of which funnel into `ingest_chunks` in `app.py`. The /chat endpoint and `query` CLI share `search_documents`.
## Project layout
- `app.py` — single-file CLI + FastAPI app. Subcommands: `ingest`, `ingest-file`, `query`, `traverse`, `serve`.
- `static/index.html` — single-file HTML/CSS/JS GUI served by `serve` (upload card + chat box + citations).
- `sample_data.json` — 8 example documents used by `ingest` (a JSON list of `{id, title, text, category}`).
- `requirements.txt` — Python deps: `qdrant-client`, `requests`, `fastapi`, `uvicorn`, `pydantic`, `pypdf`, `python-multipart`.
Key symbols inside `app.py`, if you want to read along:
- `extract_pdf_pages`, `extract_chunks_from_bytes` — file → tokens with optional page numbers.
- `chunk_words` — word-window chunker with overlap.
- `ensure_collection`, `_delete_chunks_by_source`, `ingest_chunks` — Qdrant collection + upsert plumbing.
- `ingest_bytes` — top-level helper used by both the CLI and HTTP layer.
- `create_rag_app` — FastAPI factory mounting `/chat`, `/ingest`, `/health`, and the static GUI.
## Prerequisites
1. **Python 3.9+**
2. **Qdrant** running on `http://localhost:6333`
3. **Ollama** running on `http://localhost:11434`
4. An **embedding model** pulled in Ollama (default: `nomic-embed-text`, ~270 MB)
5. A **generation model** pulled in Ollama for `serve` (default: `llama3.2`, ~2 GB)
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
This drops + recreates the collection `ollama_demo_docs` with cosine distance and ingests `sample_data.json` through the unified chunk pipeline. Output looks like:
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
or start the server and use the chat endpoint:
```bash
python app.py serve
# then, in another shell:
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What does the whitepaper say about pricing?", "limit": 3}'
```
The response includes `answer` plus `citations[]` with `source`, `page_start`/`page_end`, `chunk_index`, and a text preview — enough to render "go to page N of file X" UI.
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
### 3. Embedding (`get_embedding`)
Calls Ollama's `/api/embed` (preferred) and falls back to `/api/embeddings` for older versions. Returns a `list[float]`. The vector size is discovered lazily from the first chunk on each ingest, so swapping embedding models "just works" for a fresh collection.
Today this is sequential (one HTTP call per chunk). For larger corpora you can batch by sending an `input: [...]` list to `/api/embed`; the surface area is small enough that the change lives entirely in `ingest_chunks`.
### 4. Collection management (`ensure_collection`)
- If the collection doesn't exist, it's created with `VectorParams(size=<vector_size>, distance=Distance.COSINE)`.
- If it exists and the existing vector size differs from what the current model produces, you get a clear error pointing at the model mismatch (instead of a confusing upsert failure later).
This is also the natural place to plug in payload indexes (`client.create_payload_index(...)`) when you start filtering on `source`, `category`, `tags`, or `created_at`, and to declare sparse vectors for hybrid search.
### 5. Upsert (`ingest_chunks`)
For each chunk:
- Embed it with Ollama.
- Build a payload (see schema below).
- Compute a deterministic point id: `uuid5(namespace, f"{source}#{chunk_index}")`. This means re-ingesting the same source overwrites the previous chunks one-for-one — the upsert is idempotent.
If `replace_existing=True` (the default for `ingest-file` and `POST /ingest`), `_delete_chunks_by_source` first removes any prior chunks with the same `source` filter so renamed/shortened documents don't leave orphan chunks behind.
### 6. Re-use across entrypoints (`ingest_bytes`)
`ingest_bytes` is the single function the CLI (`ingest_files_command`) and HTTP layer (`POST /ingest`) both call. It returns an `IngestReport` with `source`, `title`, `source_type`, `chunks_ingested`, `pages`, and an optional `skipped_reason` — that's also what the HTTP response surfaces per-file. The sample-JSON path (`ingest_documents`) goes through `ingest_chunks` directly so its payload schema matches uploaded files exactly.
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
                     [--chunk-size 400] [--chunk-overlap 50]
```
Drops + recreates the collection, then ingests every document in the data file through the unified chunk pipeline. Override `--data-file` to point at any JSON list of `{id, title, text, category}` objects.
### `ingest-file` — chunk + embed local PDF / MD / TXT files
```bash
python app.py ingest-file PATH [PATH ...] \
  [--collection NAME] [--category LABEL] [--tags a,b,c] \
  [--chunk-size 400] [--chunk-overlap 50] [--no-replace]
```
- The collection is created lazily if it doesn't exist.
- Existing chunks for the same source filename are replaced by default; pass `--no-replace` to keep them.
- Supported extensions: `.pdf`, `.md`, `.markdown`, `.txt`.
- Per-file size limit: 25 MB (oversize files are skipped with a printed reason).
- Files that yield no extractable text (e.g. scanned PDFs without OCR) are skipped with a printed reason.
### `query` — semantic search
```bash
python app.py query --query "…" [--limit 3] [--collection NAME]
```
Prints the top results with score, id, title, category, and a text preview.
### `traverse` — scroll through stored points
```bash
python app.py traverse [--batch-size 4] [--limit 0] [--collection NAME]
```
Iterates through every point using Qdrant scroll pagination (no similarity required). Use `--limit N` to cap the number of points printed; `0` (default) means no limit.
### `serve` — run the FastAPI server + GUI
```bash
python app.py serve [--collection NAME] [--chat-model llama3.2] \
                    [--retrieval-limit 3] [--host 127.0.0.1] [--port 8000]
```
Mounts `/`, `/health`, `/chat`, `/ingest`, and `/docs`. See the HTTP API section below.
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
{ "message": "How do I run Qdrant locally?", "limit": 3 }
```
`limit` is optional (defaults to `--retrieval-limit`, normally `3`; max `10`).
Response:
```json
{
  "answer": "…model output grounded in the retrieved chunks…",
  "collection": "ollama_demo_docs",
  "chat_model": "llama3.2",
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
- `OLLAMA_URL` — default `http://localhost:11434`
- `OLLAMA_MODEL` — embedding model, default `nomic-embed-text`
- `OLLAMA_CHAT_MODEL` — generation model used by `serve`, default `llama3.2`
CLI flags always override env vars.
## Extending the demo
A few directions the code is structured to support:
- **Payload filtering.** The payload schema already includes `source`, `category`, `tags`, `source_type`, and `created_at`. Add `client.create_payload_index(...)` calls in `ensure_collection` and pass a `models.Filter` to `client.query_points` in `search_documents`.
- **Hybrid (dense + sparse) search.** `ensure_collection` is the single place that declares vector config; extend it to also configure a sparse vector (e.g. BM25) and emit a sparse vector per chunk inside `ingest_chunks`.
- **Batched embeddings.** Replace the per-chunk `get_embedding` loop in `ingest_chunks` with a batched call to Ollama's `/api/embed` (which accepts `input: [...]`). The rest of the pipeline doesn't need to change.
- **New file formats.** Add an extractor that returns a `List[(word, page_or_none)]` and one branch in `extract_chunks_from_bytes`.
## Troubleshooting
- **`zsh: command not found: python`** — use `python3` or the venv's interpreter (`./.venv/bin/python`).
- **`ModuleNotFoundError: No module named 'fastapi'` / `pypdf` / `multipart`** — the venv isn't active or deps aren't installed. Run `source .venv/bin/activate && pip install -r requirements.txt`, or call `./.venv/bin/python app.py …`.
- **`/ingest` returns `No extractable text found`** — the PDF is likely scanned images. Run it through OCR (e.g. `ocrmypdf input.pdf output.pdf`) and re-upload.
- **`/ingest` returns a vector-size error** — the collection was created with a different embedding model. Drop the collection (`python app.py ingest` recreates it) or switch back to the matching `--model`.
- **`/chat` returns `Unable to generate embeddings from Ollama`** — Ollama isn't running, or the embedding model isn't pulled. Check with `curl http://localhost:11434/api/tags` and `ollama list`.
- **`/chat` returns `Unable to generate a response from Ollama`** — the chat model (default `llama3.2`) isn't pulled. Run `ollama pull llama3.2`.
- **First `/chat` or `/ingest` request is slow** — Ollama needs to load models into memory on cold start; subsequent requests are much faster.
- **`urllib3 NotOpenSSLWarning` about LibreSSL** — cosmetic; safe to ignore on macOS system Python.
