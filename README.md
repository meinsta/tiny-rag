# Ollama + Qdrant Tiny RAG Demo
A small, end-to-end retrieval-augmented generation (RAG) demo. Documents are embedded with [Ollama](https://ollama.com), stored in [Qdrant](https://qdrant.tech), and retrieved at query time to ground answers from a local chat model. Includes a CLI and a lightweight browser GUI.
## Architecture
```
  user ──▶ GUI / curl ──▶ FastAPI (/chat)
                          │
                          ├─ embed query    ──▶ Ollama (nomic-embed-text)
                          ├─ vector search ──▶ Qdrant (ollama_demo_docs)
                          └─ generate      ──▶ Ollama (llama3.2)
```
## Project layout
- `app.py` — CLI entrypoint with `ingest`, `query`, `traverse`, and `serve` subcommands
- `static/index.html` — single-file HTML/CSS/JS GUI served by `serve`
- `sample_data.json` — 8 example documents used by `ingest`
- `requirements.txt` — Python dependencies (`qdrant-client`, `requests`, `fastapi`, `uvicorn`, `pydantic`)
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
## Usage
All commands accept the global flags `--qdrant-url`, `--ollama-url`, and `--model` (embedding model). Subcommands add their own.
### 1. Ingest sample data
```bash
python app.py ingest
```
Recreates the collection `ollama_demo_docs` with cosine distance and writes all 8 sample documents (each embedded via Ollama). Override the source file with `--data-file path/to/file.json` (must be a JSON list of `{id, title, text, category}` objects).
### 2. Semantic query
```bash
python app.py query --query "How do I run Qdrant locally?" --limit 3
```
Prints the top results with score, id, title, category, and a text preview.
### 3. Traverse stored points
```bash
python app.py traverse --batch-size 3
```
Iterates through every point using Qdrant scroll pagination (no similarity required). Use `--limit N` to cap the number of points printed.
### 4. Run the RAG chat endpoint + GUI
```bash
python app.py serve
```
Starts FastAPI/uvicorn on `http://127.0.0.1:8000` and exposes:
- `GET /` — browser GUI (question box, retrieval-limit input, answer + citations, live health pill)
- `GET /health` — liveness probe (`{"status":"ok"}`)
- `POST /chat` — JSON body `{"message": "…", "limit": 3}`; returns `{answer, collection, chat_model, citations:[…]}`
- `GET /docs` — auto-generated OpenAPI/Swagger UI
Open <http://127.0.0.1:8000> in a browser to use the GUI. Tip: ⌘/Ctrl+Enter sends from the textarea.
Example request:
```bash
curl -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "How do I run Qdrant locally?", "limit": 3}'
```
Useful `serve` flags:
- `--collection` (default: `ollama_demo_docs`)
- `--chat-model` (default: `llama3.2`)
- `--retrieval-limit` (default: `3`) — used when the request body omits `limit`
- `--host` (default: `127.0.0.1`), `--port` (default: `8000`)
## Configuration via environment variables
- `QDRANT_URL` — default `http://localhost:6333`
- `OLLAMA_URL` — default `http://localhost:11434`
- `OLLAMA_MODEL` — embedding model, default `nomic-embed-text`
- `OLLAMA_CHAT_MODEL` — generation model used by `serve`, default `llama3.2`
CLI flags always override env vars.
## Troubleshooting
- **`zsh: command not found: python`** — use `python3` or the venv's interpreter (`./.venv/bin/python`).
- **`ModuleNotFoundError: No module named 'fastapi'`** — the venv isn't active. Run `source .venv/bin/activate` or call `./.venv/bin/python app.py …`.
- **`/chat` returns `Unable to generate embeddings from Ollama`** — Ollama isn't running, or the embedding model isn't pulled. Check with `curl http://localhost:11434/api/tags` and `ollama list`.
- **`/chat` returns `Unable to generate a response from Ollama`** — the chat model (default `llama3.2`) isn't pulled. Run `ollama pull llama3.2`.
- **First `/chat` request is slow** — Ollama needs to load the chat model into memory on cold start; subsequent requests are much faster.
- **`urllib3 NotOpenSSLWarning` about LibreSSL** — cosmetic; safe to ignore on macOS system Python.
