from __future__ import annotations

import argparse
import io
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import requests
from qdrant_client import QdrantClient, models

DEFAULT_COLLECTION = "ollama_demo_docs"
DEFAULT_DATA_FILE = Path(__file__).with_name("sample_data.json")
STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

# Chunking defaults. Word-based windowing keeps the demo dependency-light
# (no tokenizer required) while still producing reasonable chunks for the
# default embedding model.
DEFAULT_CHUNK_WORDS = 400
DEFAULT_CHUNK_OVERLAP = 50

# Stable namespace for deterministic point IDs derived from (source, chunk_index).
# Re-ingesting the same source+chunk_index overwrites the previous point.
_POINT_ID_NAMESPACE = uuid.UUID("6f1d2c3a-9b0e-4a8e-9f31-1c2b3d4e5f60")

SUPPORTED_FILE_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB per file; demo guardrail.

# Module-level cache for the sparse embedding model (lazy-loaded on first use).
_sparse_encoder: Optional[Any] = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    limit: Optional[int] = Field(default=None, ge=1, le=10)
    # Optional payload filters — any combination narrows the search scope.
    filter_category: Optional[str] = None
    filter_source: Optional[str] = None
    filter_source_type: Optional[str] = None
    filter_tags: Optional[List[str]] = None


def load_documents(data_file: Path) -> List[Dict[str, Any]]:
    with data_file.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of documents in {data_file}")

    required_fields = {"id", "title", "text", "category"}
    documents: List[Dict[str, Any]] = []
    for index, doc in enumerate(raw):
        if not isinstance(doc, dict):
            raise ValueError(f"Document at index {index} is not an object")

        missing = required_fields.difference(doc.keys())
        if missing:
            raise ValueError(
                f"Document at index {index} is missing required fields: {sorted(missing)}"
            )

        documents.append(
            {
                "id": int(doc["id"]),
                "title": str(doc["title"]),
                "text": str(doc["text"]),
                "category": str(doc["category"]),
            }
        )
    return documents


def _extract_embedding(data: Dict[str, Any]) -> Optional[List[float]]:
    direct = data.get("embedding")
    if isinstance(direct, list) and direct and isinstance(direct[0], (float, int)):
        return [float(value) for value in direct]

    nested = data.get("embeddings")
    if isinstance(nested, list) and nested:
        first = nested[0]
        if isinstance(first, list) and first and isinstance(first[0], (float, int)):
            return [float(value) for value in first]
        if isinstance(first, (float, int)):
            return [float(value) for value in nested]

    return None


def get_embedding(text: str, model: str, ollama_url: str) -> List[float]:
    base_url = ollama_url.rstrip("/")
    attempts = [
        (f"{base_url}/api/embed", {"model": model, "input": text}),
        (f"{base_url}/api/embeddings", {"model": model, "prompt": text}),
    ]

    errors: List[str] = []
    for url, payload in attempts:
        try:
            response = requests.post(url, json=payload, timeout=120)
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue

        if response.status_code in (404, 405):
            errors.append(f"{url}: endpoint unavailable ({response.status_code})")
            continue

        try:
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue
        except ValueError as exc:
            errors.append(f"{url}: invalid JSON response ({exc})")
            continue

        embedding = _extract_embedding(data)
        if embedding:
            return embedding
        errors.append(f"{url}: response did not include an embedding vector")

    raise RuntimeError(
        "Unable to generate embeddings from Ollama. "
        "Ensure Ollama is running and the embedding model is available. "
        f"Details: {'; '.join(errors)}"
    )

def generate_text(prompt: str, model: str, ollama_url: str) -> str:
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, timeout=240)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Unable to generate a response from Ollama. "
            "Ensure Ollama is running and the generation model is available. "
            f"Details: {exc}"
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Ollama generation endpoint returned invalid JSON: {exc}") from exc

    text = data.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise RuntimeError("Ollama generation response did not include a non-empty 'response' field.")


def _sparse_available() -> bool:
    """Return True if fastembed is installed and the sparse encoder can be used."""
    try:
        from fastembed import SparseTextEmbedding  # noqa: F401
        return True
    except ImportError:
        return False


def _get_sparse_encoder() -> Any:
    """Lazy-load and cache the BM42 sparse text encoder."""
    global _sparse_encoder
    if _sparse_encoder is None:
        from fastembed import SparseTextEmbedding
        _sparse_encoder = SparseTextEmbedding(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
        )
    return _sparse_encoder


def get_sparse_embedding(text: str) -> Optional[models.SparseVector]:
    """Generate a BM42 sparse vector for *text*, or return None if unavailable."""
    if not _sparse_available():
        return None
    try:
        encoder = _get_sparse_encoder()
        result = next(iter(encoder.embed([text])))
        return models.SparseVector(
            indices=list(result.indices),
            values=list(result.values),
        )
    except Exception:
        return None


def format_preview(text: str, max_length: int = 120) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 3]}..."


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

# CLI mode names. "none" disables quantization; the rest map onto Qdrant's
# scalar (int8), binary (1 bit/dim), and product (PQ x16) configs.
QUANTIZATION_MODES: Tuple[str, ...] = ("none", "scalar", "binary", "product")


def build_quantization_config(
    mode: str, *, always_ram: bool = True
) -> Optional[Any]:
    """Translate a CLI mode string into a Qdrant quantization config object.

    Returns None for ``mode='none'``. Defaults follow Qdrant's docs:
      - scalar:  int8 with quantile=0.99 (good general default)
      - binary:  1 bit per dimension (works best on >=1024-dim embeddings)
      - product: x16 compression (largest savings, lowest recall)

    ``always_ram=True`` keeps the *quantized* vectors resident in RAM so queries
    stay fast even when the original full-precision vectors live on disk.
    """
    mode_norm = (mode or "none").lower()
    if mode_norm == "none":
        return None
    if mode_norm == "scalar":
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=always_ram,
            )
        )
    if mode_norm == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=always_ram)
        )
    if mode_norm == "product":
        return models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=models.CompressionRatio.X16,
                always_ram=always_ram,
            )
        )
    raise ValueError(
        f"Unknown quantization mode '{mode}'. "
        f"Choose from: {', '.join(QUANTIZATION_MODES)}"
    )


def detect_quantization_mode(info: Any) -> str:
    """Inspect a CollectionInfo object and return one of QUANTIZATION_MODES."""
    qcfg = getattr(getattr(info, "config", None), "quantization_config", None)
    if qcfg is None:
        return "none"
    if getattr(qcfg, "scalar", None) is not None:
        return "scalar"
    if getattr(qcfg, "binary", None) is not None:
        return "binary"
    if getattr(qcfg, "product", None) is not None:
        return "product"
    return "none"


def estimate_vector_bytes(num_points: int, dim: int, mode: str) -> int:
    """Lower-bound estimate of dense-vector storage in bytes.

    Sparse vectors, HNSW graph links, payload indexes, and segment overhead
    are NOT included — the goal is a clean apples-to-apples comparison of the
    raw vector storage cost across quantization modes.
    """
    if dim < 0 or num_points < 0:
        raise ValueError("num_points and dim must be non-negative")
    if mode == "none":
        return num_points * dim * 4  # float32
    if mode == "scalar":
        return num_points * dim  # int8
    if mode == "binary":
        return num_points * ((dim + 7) // 8)  # 1 bit per dim, byte-aligned
    if mode == "product":
        return (num_points * dim * 4) // 16  # X16 compression
    raise ValueError(f"Unknown mode: {mode}")


def format_bytes(n: float) -> str:
    """Render a byte count using the largest unit that yields a value < 1024."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def build_quantization_search_params(
    rescore: Optional[bool], oversampling: Optional[float]
) -> Optional[models.SearchParams]:
    """Construct a SearchParams object that tunes quantized re-scoring.

    Returns None when neither knob is provided so callers can omit the
    ``search_params`` argument entirely on collections without quantization.
    """
    if rescore is None and oversampling is None:
        return None
    return models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore if rescore is not None else True,
            oversampling=oversampling if oversampling is not None else 1.0,
        )
    )


# ---------------------------------------------------------------------------
# Extraction + chunking pipeline
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single chunk of text ready to be embedded and upserted."""

    text: str
    chunk_index: int
    char_start: int
    char_end: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_count: int = 0  # populated once the full chunk list is known


@dataclass
class IngestReport:
    """Summary of an ingest operation for a single source."""

    source: str
    title: str
    source_type: str
    chunks_ingested: int
    pages: Optional[int] = None
    skipped_reason: Optional[str] = None


def extract_pdf_pages(data: bytes) -> List[Tuple[int, str]]:
    """Extract per-page text from a PDF byte string. Pages are 1-indexed."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "pypdf is required for PDF ingestion. Install dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Could not parse PDF: {exc}") from exc

    pages: List[Tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((index, text))
    return pages


def _words_with_pages_from_text(text: str) -> List[Tuple[str, Optional[int]]]:
    return [(token, None) for token in text.split()]


def _words_with_pages_from_pdf(
    pages: Sequence[Tuple[int, str]],
) -> List[Tuple[str, Optional[int]]]:
    out: List[Tuple[str, Optional[int]]] = []
    for page_number, page_text in pages:
        for token in page_text.split():
            out.append((token, page_number))
    return out


def chunk_words(
    words_with_pages: Sequence[Tuple[str, Optional[int]]],
    *,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split a token stream into overlapping windows.

    The token stream pairs each whitespace-delimited token with an optional
    page number (used for PDFs). Char offsets are computed against the joined
    " ".join(...) representation so they line up with the chunk text we store.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    total_words = len(words_with_pages)
    if total_words == 0:
        return []

    word_strings = [w for w, _ in words_with_pages]
    page_per_word = [p for _, p in words_with_pages]

    # Char offset of each word in the eventual joined text (single-space separator).
    char_offsets: List[int] = []
    cursor = 0
    for word in word_strings:
        char_offsets.append(cursor)
        cursor += len(word) + 1  # word + separating space

    step = chunk_size - chunk_overlap
    chunks: List[Chunk] = []
    start = 0
    while start < total_words:
        end = min(start + chunk_size, total_words)
        text = " ".join(word_strings[start:end])
        char_start = char_offsets[start]
        char_end = char_offsets[end - 1] + len(word_strings[end - 1])
        sub_pages = [p for p in page_per_word[start:end] if p is not None]
        page_start = min(sub_pages) if sub_pages else None
        page_end = max(sub_pages) if sub_pages else None
        chunks.append(
            Chunk(
                text=text,
                chunk_index=len(chunks),
                char_start=char_start,
                char_end=char_end,
                page_start=page_start,
                page_end=page_end,
            )
        )
        if end == total_words:
            break
        start += step

    total_chunks = len(chunks)
    for chunk in chunks:
        chunk.chunk_count = total_chunks
    return chunks


def extract_chunks_from_bytes(
    *,
    filename: str,
    data: bytes,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Tuple[List[Chunk], str, Optional[int]]:
    """Parse raw bytes by file extension and return chunks plus source_type.

    Returns (chunks, source_type, page_count). page_count is only set for PDFs.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix or '(none)'}'. Supported: "
            f"{sorted(SUPPORTED_FILE_SUFFIXES)}"
        )

    page_count: Optional[int] = None
    if suffix == ".pdf":
        pages = extract_pdf_pages(data)
        page_count = len(pages)
        words = _words_with_pages_from_pdf(pages)
        source_type = "pdf"
    elif suffix in (".md", ".markdown"):
        text = data.decode("utf-8", errors="replace")
        words = _words_with_pages_from_text(text)
        source_type = "markdown"
    else:  # .txt
        text = data.decode("utf-8", errors="replace")
        words = _words_with_pages_from_text(text)
        source_type = "text"

    chunks = chunk_words(words, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks, source_type, page_count


# ---------------------------------------------------------------------------
# Shared ingest pipeline (used by CLI and HTTP endpoints)
# ---------------------------------------------------------------------------


def _deterministic_point_id(source: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{source}#{chunk_index}"))


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    *,
    quantization_config: Optional[Any] = None,
) -> None:
    """Create the collection if missing; verify vector size if it already exists.

    ``quantization_config`` is applied only when the collection is being created.
    For existing collections, use ``python app.py quantize`` to update it in
    place — a warning is printed when a quantization config is supplied but
    ignored, so SAs don't silently keep the old setting.
    """
    if client.collection_exists(collection):
        info = client.get_collection(collection)
        vectors_cfg = info.config.params.vectors

        # Detect legacy unnamed-vector schema created by an older version.
        if not isinstance(vectors_cfg, dict):
            raise RuntimeError(
                f"Collection '{collection}' was created with an older unnamed-vector "
                "schema. Run 'python app.py ingest' to drop and recreate it."
            )

        dense_cfg = vectors_cfg.get("dense")
        existing_size = getattr(dense_cfg, "size", None) if dense_cfg else None
        if existing_size is not None and existing_size != vector_size:
            raise RuntimeError(
                f"Collection '{collection}' was created with vector size "
                f"{existing_size}, but the current embedding model produces "
                f"vectors of size {vector_size}. Drop the collection or pick a "
                "matching model before ingesting."
            )

        if quantization_config is not None:
            current_mode = detect_quantization_mode(info)
            print(
                f"  note: collection '{collection}' already exists with "
                f"quantization='{current_mode}'. Ignoring --quantization for "
                "this ingest. Run 'python app.py quantize --mode <mode>' to "
                "change it in place."
            )
        return

    sparse_config = (
        {"sparse": models.SparseVectorParams()} if _sparse_available() else None
    )
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config=sparse_config,
        quantization_config=quantization_config,
    )

    # Keyword payload indexes for efficient filtered search.
    for field in ("category", "source", "source_type", "tags"):
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema="keyword",
        )


def _delete_chunks_by_source(
    client: QdrantClient, collection: str, source: str
) -> None:
    if not client.collection_exists(collection):
        return
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source", match=models.MatchValue(value=source)
                    )
                ]
            )
        ),
    )


def ingest_chunks(
    *,
    client: QdrantClient,
    collection: str,
    embed_model: str,
    ollama_url: str,
    title: str,
    source: str,
    source_type: str,
    category: str,
    tags: Sequence[str],
    chunks: Sequence[Chunk],
    replace_existing: bool,
    extra_payload: Optional[Dict[str, Any]] = None,
    quantization_config: Optional[Any] = None,
) -> int:
    """Embed chunks via Ollama and upsert them into Qdrant with rich payload.

    Returns the number of points successfully upserted.
    """
    if not chunks:
        return 0

    # Probe vector size from the first chunk so we can lazily create the
    # collection on the very first ingest call.
    first_vector = get_embedding(chunks[0].text, embed_model, ollama_url)
    ensure_collection(
        client,
        collection,
        len(first_vector),
        quantization_config=quantization_config,
    )

    if replace_existing:
        _delete_chunks_by_source(client, collection, source)

    created_at = datetime.now(timezone.utc).isoformat()
    tag_list = [str(t) for t in (tags or [])]
    points: List[models.PointStruct] = []
    for idx, chunk in enumerate(chunks):
        vector = first_vector if idx == 0 else get_embedding(
            chunk.text, embed_model, ollama_url
        )
        payload: Dict[str, Any] = {
            "title": title,
            "text": chunk.text,
            "category": category,
            "source": source,
            "source_type": source_type,
            "chunk_index": chunk.chunk_index,
            "chunk_count": chunk.chunk_count,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "tags": tag_list,
            "created_at": created_at,
        }
        if chunk.page_start is not None:
            payload["page_start"] = chunk.page_start
            payload["page_end"] = chunk.page_end
        if extra_payload:
            payload.update(extra_payload)
        chunk_vectors: Dict[str, Any] = {"dense": vector}
        sparse_vec = get_sparse_embedding(chunk.text)
        if sparse_vec is not None:
            chunk_vectors["sparse"] = sparse_vec
        points.append(
            models.PointStruct(
                id=_deterministic_point_id(source, chunk.chunk_index),
                vector=chunk_vectors,
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points)
    return len(points)


def ingest_bytes(
    *,
    client: QdrantClient,
    collection: str,
    embed_model: str,
    ollama_url: str,
    filename: str,
    data: bytes,
    category: str,
    tags: Sequence[str],
    replace_existing: bool,
    chunk_size: int = DEFAULT_CHUNK_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    quantization_config: Optional[Any] = None,
) -> IngestReport:
    """Top-level helper: bytes -> chunks -> embed -> upsert. Used by HTTP + CLI."""
    chunks, source_type, page_count = extract_chunks_from_bytes(
        filename=filename,
        data=data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    title = Path(filename).stem or filename
    source = Path(filename).name or filename

    if not chunks:
        return IngestReport(
            source=source,
            title=title,
            source_type=source_type,
            chunks_ingested=0,
            pages=page_count,
            skipped_reason=(
                "No extractable text found. Scanned PDFs require OCR before "
                "ingestion."
            ),
        )

    ingested = ingest_chunks(
        client=client,
        collection=collection,
        embed_model=embed_model,
        ollama_url=ollama_url,
        title=title,
        source=source,
        source_type=source_type,
        category=category,
        tags=tags,
        chunks=chunks,
        replace_existing=replace_existing,
        extra_payload={"page_count": page_count} if page_count is not None else None,
        quantization_config=quantization_config,
    )
    return IngestReport(
        source=source,
        title=title,
        source_type=source_type,
        chunks_ingested=ingested,
        pages=page_count,
    )


def build_filter(
    *,
    category: Optional[str] = None,
    source: Optional[str] = None,
    source_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Optional[models.Filter]:
    """Build a Qdrant payload filter from optional keyword constraints.

    All provided constraints are combined with AND (must). Tags are each
    added as a separate must-condition so every supplied tag must be present.
    """
    conditions: List[Any] = []
    if category:
        conditions.append(
            models.FieldCondition(key="category", match=models.MatchValue(value=category))
        )
    if source:
        conditions.append(
            models.FieldCondition(key="source", match=models.MatchValue(value=source))
        )
    if source_type:
        conditions.append(
            models.FieldCondition(
                key="source_type", match=models.MatchValue(value=source_type)
            )
        )
    if tags:
        for tag in tags:
            conditions.append(
                models.FieldCondition(key="tags", match=models.MatchValue(value=tag))
            )
    return models.Filter(must=conditions) if conditions else None


def search_documents(
    query: str,
    *,
    qdrant_url: str,
    ollama_url: str,
    collection: str,
    model: str,
    limit: int,
    filter_category: Optional[str] = None,
    filter_source: Optional[str] = None,
    filter_source_type: Optional[str] = None,
    filter_tags: Optional[List[str]] = None,
    rescore: Optional[bool] = None,
    oversampling: Optional[float] = None,
) -> Tuple[List[Any], str]:
    """Return (points, search_mode).

    search_mode is 'hybrid' when dense and sparse vectors are fused via RRF,
    or 'dense' when only the dense vector is used.

    ``rescore`` and ``oversampling`` are passed through to Qdrant's quantization
    search params. They have no effect on collections without quantization but
    are safe to set unconditionally — use them to demo recall vs latency on a
    quantized collection.
    """
    client = QdrantClient(url=qdrant_url)
    dense_vector = get_embedding(query, model, ollama_url)
    query_filter = build_filter(
        category=filter_category,
        source=filter_source,
        source_type=filter_source_type,
        tags=filter_tags,
    )
    search_params = build_quantization_search_params(rescore, oversampling)

    # Inspect the collection's vector configuration.
    has_sparse = False
    is_legacy = False
    try:
        info = client.get_collection(collection)
        vectors_cfg = info.config.params.vectors
        is_legacy = not isinstance(vectors_cfg, dict)
        if not is_legacy:
            sv = getattr(info.config.params, "sparse_vectors", None)
            has_sparse = bool(sv and "sparse" in sv)
    except Exception:
        pass

    if is_legacy:
        # Older unnamed-vector schema: fall back to unnamespaced query.
        response = client.query_points(
            collection_name=collection,
            query=dense_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
            search_params=search_params,
        )
        return response.points, "dense"

    if has_sparse:
        sparse_vec = get_sparse_embedding(query)
    else:
        sparse_vec = None

    if sparse_vec is not None:
        # Hybrid: prefetch from both indices, then fuse with RRF.
        # Quantization tuning only applies to the dense prefetch — sparse
        # vectors are not quantized.
        prefetch_limit = max(limit * 4, 20)
        response = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_limit,
                    filter=query_filter,
                    params=search_params,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points, "hybrid"

    # Named-vector dense-only search.
    response = client.query_points(
        collection_name=collection,
        query=dense_vector,
        using="dense",
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
        search_params=search_params,
    )
    return response.points, "dense"


def build_rag_prompt(message: str, contexts: List[Dict[str, Any]]) -> str:
    if contexts:
        formatted_context = "\n\n".join(
            [
                (
                    f"[{index}] title={item['title']} category={item['category']} score={item['score']:.4f}\n"
                    f"{item['text']}"
                )
                for index, item in enumerate(contexts, start=1)
            ]
        )
    else:
        formatted_context = "No relevant context was retrieved."

    return (
        "You are a concise assistant. Answer the user using the provided context.\n"
        "If the context is insufficient, say so clearly.\n\n"
        f"Context:\n{formatted_context}\n\n"
        f"User question: {message}\n"
        "Answer:"
    )


def ingest_documents(args: argparse.Namespace) -> None:
    """Ingest the bundled JSON sample data through the unified chunk pipeline.

    The collection is dropped and recreated to mirror the original behavior:
    'ingest' is a clean-slate command for the canned sample dataset.
    """
    documents = load_documents(Path(args.data_file))
    if not documents:
        raise ValueError("No documents to ingest")

    client = QdrantClient(url=args.qdrant_url)
    if client.collection_exists(args.collection):
        client.delete_collection(args.collection)

    quantization_mode = getattr(args, "quantization", "none")
    quantization_config = build_quantization_config(
        quantization_mode, always_ram=getattr(args, "always_ram", True)
    )

    total_chunks = 0
    for doc in documents:
        body = f"{doc['title']}\n{doc['text']}"
        words = _words_with_pages_from_text(body)
        chunks = chunk_words(
            words,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        if not chunks:
            continue
        source = f"sample:{doc['id']}"
        total_chunks += ingest_chunks(
            client=client,
            collection=args.collection,
            embed_model=args.model,
            ollama_url=args.ollama_url,
            title=doc["title"],
            source=source,
            source_type="sample",
            category=doc["category"],
            tags=[],
            chunks=chunks,
            replace_existing=False,  # collection was just dropped
            extra_payload={"sample_id": doc["id"]},
            # Only the first ingest call actually creates the collection;
            # subsequent calls hit the existing-collection branch in
            # ensure_collection and silently ignore quantization_config.
            quantization_config=quantization_config,
        )

    print(
        f"Ingested {total_chunks} chunks across {len(documents)} sample documents "
        f"into collection '{args.collection}' using model '{args.model}' "
        f"(quantization={quantization_mode})."
    )


def ingest_files_command(args: argparse.Namespace) -> None:
    """CLI entrypoint for ingesting one or more local files (PDF/MD/TXT)."""
    paths = [Path(p) for p in args.paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"File(s) not found: {', '.join(missing)}")

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    client = QdrantClient(url=args.qdrant_url)
    quantization_config = build_quantization_config(
        getattr(args, "quantization", "none"),
        always_ram=getattr(args, "always_ram", True),
    )

    total_chunks = 0
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FILE_SUFFIXES:
            print(f"  skipped {path.name}: unsupported extension '{suffix}'")
            continue
        data = path.read_bytes()
        if len(data) > MAX_UPLOAD_BYTES:
            print(
                f"  skipped {path.name}: file is {len(data)} bytes, "
                f"exceeds {MAX_UPLOAD_BYTES} byte limit"
            )
            continue
        try:
            report = ingest_bytes(
                client=client,
                collection=args.collection,
                embed_model=args.model,
                ollama_url=args.ollama_url,
                filename=path.name,
                data=data,
                category=args.category,
                tags=tags,
                replace_existing=args.replace,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                quantization_config=quantization_config,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"  failed {path.name}: {exc}")
            continue

        if report.skipped_reason:
            print(f"  skipped {path.name}: {report.skipped_reason}")
            continue
        page_info = f", {report.pages} pages" if report.pages else ""
        print(
            f"  {path.name} ({report.source_type}{page_info}): "
            f"{report.chunks_ingested} chunks"
        )
        total_chunks += report.chunks_ingested

    print(
        f"Ingested {total_chunks} chunks total into collection '{args.collection}'."
    )


def query_documents(args: argparse.Namespace) -> None:
    filter_tags = (
        [t.strip() for t in args.filter_tags.split(",") if t.strip()]
        if args.filter_tags
        else None
    )
    results, search_mode = search_documents(
        args.query,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        collection=args.collection,
        model=args.model,
        limit=args.limit,
        filter_category=args.filter_category,
        filter_source=args.filter_source,
        filter_source_type=args.filter_source_type,
        filter_tags=filter_tags,
        rescore=getattr(args, "rescore", None),
        oversampling=getattr(args, "oversampling", None),
    )

    if not results:
        print("No results found.")
        return

    print(f"Search mode: {search_mode}")
    for index, result in enumerate(results, start=1):
        payload = result.payload or {}
        title = payload.get("title", "(untitled)")
        category = payload.get("category", "unknown")
        preview = format_preview(str(payload.get("text", "")))
        print(f"{index}. score={result.score:.4f} id={result.id}")
        print(f"   title: {title}")
        print(f"   category: {category}")
        print(f"   text: {preview}")


def memory_report(args: argparse.Namespace) -> None:
    """Print a side-by-side memory estimate across all quantization modes.

    Useful for SAs demoing the quantization tradeoff: same collection, four
    storage profiles, the active one called out so the customer sees what
    they're getting today vs. what they could get.
    """
    client = QdrantClient(url=args.qdrant_url)
    if not client.collection_exists(args.collection):
        raise RuntimeError(
            f"Collection '{args.collection}' does not exist. Ingest first."
        )

    info = client.get_collection(args.collection)
    points = client.count(args.collection, exact=True).count

    vectors_cfg = info.config.params.vectors
    if isinstance(vectors_cfg, dict):
        dense_cfg = vectors_cfg.get("dense")
    else:
        dense_cfg = vectors_cfg
    dim = getattr(dense_cfg, "size", None) if dense_cfg else None
    distance = getattr(dense_cfg, "distance", None) if dense_cfg else None

    has_sparse = bool(getattr(info.config.params, "sparse_vectors", None))
    active_mode = detect_quantization_mode(info)

    print(f"Collection: {args.collection}")
    print(f"  status:               {info.status}")
    print(f"  segments:             {info.segments_count}")
    print(f"  points:               {points}")
    print(f"  dense vector size:    {dim}")
    print(f"  distance:             {distance}")
    print(f"  sparse vectors:       {'yes' if has_sparse else 'no'}")
    print(f"  active quantization:  {active_mode}")

    if not dim or not points:
        print("\nSkipping memory estimates (collection has no vectors yet).")
        return

    full_bytes = estimate_vector_bytes(points, dim, "none")
    print()
    print("Estimated dense-vector RAM by mode (lower bound, vectors only):")
    print(f"  {'mode':8s}  {'size':>10s}   ratio vs full")
    for mode in QUANTIZATION_MODES:
        size = estimate_vector_bytes(points, dim, mode)
        marker = "  <- active" if mode == active_mode else ""
        ratio = full_bytes / size if size else float("inf")
        print(
            f"  {mode:8s}  {format_bytes(size):>10s}   {ratio:5.1f}x{marker}"
        )

    if active_mode != "none":
        active_bytes = estimate_vector_bytes(points, dim, active_mode)
        saved = full_bytes - active_bytes
        pct = (saved / full_bytes) * 100 if full_bytes else 0
        print()
        print(
            f"Active quantization saves {format_bytes(saved)} "
            f"({pct:.1f}% vs full precision)."
        )

    print()
    print(
        "Note: estimates cover dense quantized vectors only. Original "
        "full-precision vectors and HNSW graph links are stored separately."
    )


def quantize_collection(args: argparse.Namespace) -> None:
    """Apply quantization to an existing collection in-place via update_collection.

    Qdrant re-quantizes existing segments asynchronously; running ``memory``
    a few seconds later is a reliable way to verify the change took effect.
    """
    client = QdrantClient(url=args.qdrant_url)
    if not client.collection_exists(args.collection):
        raise RuntimeError(
            f"Collection '{args.collection}' does not exist. Ingest first."
        )

    if args.mode == "none":
        raise RuntimeError(
            "Disabling quantization in-place is not supported by this CLI. "
            "Recreate the collection via 'python app.py ingest' (with no "
            "--quantization flag) to clear it."
        )

    qcfg = build_quantization_config(args.mode, always_ram=args.always_ram)
    client.update_collection(
        collection_name=args.collection,
        quantization_config=qcfg,
    )
    print(
        f"Applied {args.mode} quantization (always_ram={args.always_ram}) "
        f"to collection '{args.collection}'. Re-quantization runs in the "
        "background; rerun 'python app.py memory' to verify."
    )


def traverse_documents(args: argparse.Namespace) -> None:
    client = QdrantClient(url=args.qdrant_url)
    offset = None
    shown = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=args.collection,
            limit=args.batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            title = payload.get("title", "(untitled)")
            category = payload.get("category", "unknown")
            preview = format_preview(str(payload.get("text", "")))
            shown += 1
            print(f"{shown}. id={point.id} category={category} title={title}")
            print(f"   text: {preview}")

            if args.limit and shown >= args.limit:
                print(f"\nDisplayed {shown} points (limit reached).")
                return

        if next_offset is None:
            break
        offset = next_offset

    print(f"\nDisplayed {shown} points in total.")


def create_rag_app(
    *,
    qdrant_url: str,
    ollama_url: str,
    collection: str,
    embed_model: str,
    chat_model: str,
    default_limit: int,
) -> FastAPI:
    app = FastAPI(title="Tiny RAG Chat API", version="0.1.0")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(request: ChatRequest) -> Dict[str, Any]:
        effective_limit = request.limit if request.limit is not None else default_limit
        try:
            results, search_mode = search_documents(
                request.message,
                qdrant_url=qdrant_url,
                ollama_url=ollama_url,
                collection=collection,
                model=embed_model,
                limit=effective_limit,
                filter_category=request.filter_category,
                filter_source=request.filter_source,
                filter_source_type=request.filter_source_type,
                filter_tags=request.filter_tags,
            )
            contexts: List[Dict[str, Any]] = []
            for result in results:
                payload = result.payload or {}
                contexts.append(
                    {
                        "id": result.id,
                        "score": float(result.score),
                        "title": str(payload.get("title", "(untitled)")),
                        "category": str(payload.get("category", "unknown")),
                        "text": str(payload.get("text", "")),
                        "source": payload.get("source"),
                        "source_type": payload.get("source_type"),
                        "chunk_index": payload.get("chunk_index"),
                        "chunk_count": payload.get("chunk_count"),
                        "page_start": payload.get("page_start"),
                        "page_end": payload.get("page_end"),
                    }
                )

            prompt = build_rag_prompt(request.message, contexts)
            answer = generate_text(prompt, chat_model, ollama_url)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "answer": answer,
            "collection": collection,
            "chat_model": chat_model,
            "search_mode": search_mode,
            "citations": [
                {
                    "id": item["id"],
                    "score": item["score"],
                    "title": item["title"],
                    "category": item["category"],
                    "source": item["source"],
                    "source_type": item["source_type"],
                    "chunk_index": item["chunk_index"],
                    "chunk_count": item["chunk_count"],
                    "page_start": item["page_start"],
                    "page_end": item["page_end"],
                    "text_preview": format_preview(item["text"], max_length=180),
                }
                for item in contexts
            ],
        }

    @app.post("/ingest")
    async def ingest_endpoint(
        files: List[UploadFile] = File(..., description="PDF, MD, or TXT files"),
        category: str = Form("uploaded"),
        tags: str = Form("", description="Comma-separated tag list"),
        replace: bool = Form(True, description="Replace existing chunks for the same source"),
        chunk_size: int = Form(DEFAULT_CHUNK_WORDS, ge=1, le=4000),
        chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP, ge=0, le=2000),
    ) -> Dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="No files were uploaded.")
        if chunk_overlap >= chunk_size:
            raise HTTPException(
                status_code=400,
                detail="chunk_overlap must be strictly less than chunk_size.",
            )

        parsed_tags = [t.strip() for t in (tags or "").split(",") if t.strip()]
        client = QdrantClient(url=qdrant_url)
        results: List[Dict[str, Any]] = []
        total = 0
        for upload in files:
            filename = upload.filename or "upload"
            data = await upload.read()
            if len(data) > MAX_UPLOAD_BYTES:
                results.append(
                    {
                        "source": filename,
                        "chunks_ingested": 0,
                        "skipped_reason": (
                            f"File is {len(data)} bytes; exceeds limit of "
                            f"{MAX_UPLOAD_BYTES} bytes."
                        ),
                    }
                )
                continue
            try:
                report = ingest_bytes(
                    client=client,
                    collection=collection,
                    embed_model=embed_model,
                    ollama_url=ollama_url,
                    filename=filename,
                    data=data,
                    category=category,
                    tags=parsed_tags,
                    replace_existing=replace,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except ValueError as exc:
                # Bad input (unsupported type, malformed PDF). Surface per-file.
                results.append(
                    {
                        "source": filename,
                        "chunks_ingested": 0,
                        "skipped_reason": str(exc),
                    }
                )
                continue
            except RuntimeError as exc:
                # Embedding/Qdrant failures abort the whole request.
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            results.append(
                {
                    "source": report.source,
                    "title": report.title,
                    "source_type": report.source_type,
                    "pages": report.pages,
                    "chunks_ingested": report.chunks_ingested,
                    "skipped_reason": report.skipped_reason,
                }
            )
            total += report.chunks_ingested

        return {
            "collection": collection,
            "embed_model": embed_model,
            "total_chunks": total,
            "results": results,
        }

    return app


def serve_chat_endpoint(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Uvicorn is required to run the API server. Install dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc

    if _sparse_available():
        print("Hybrid search: enabled (fastembed BM42 detected).")
    else:
        print("Hybrid search: disabled (install fastembed to enable).")

    app = create_rag_app(
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        collection=args.collection,
        embed_model=args.model,
        chat_model=args.chat_model,
        default_limit=args.retrieval_limit,
    )
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tiny RAG — a small Ollama + Qdrant demo with ingest/query/traverse/serve commands."
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Create/recreate collection and ingest sample documents."
    )
    ingest_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    ingest_parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help="Words per chunk (default: %(default)s).",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Words of overlap between consecutive chunks (default: %(default)s).",
    )
    ingest_parser.add_argument(
        "--quantization",
        choices=list(QUANTIZATION_MODES),
        default="none",
        help=(
            "Quantization mode applied when (re)creating the collection "
            "(default: %(default)s). Use 'memory' afterwards to compare RAM costs."
        ),
    )
    ingest_parser.add_argument(
        "--always-ram",
        dest="always_ram",
        action="store_true",
        default=True,
        help="Keep quantized vectors in RAM (default).",
    )
    ingest_parser.add_argument(
        "--no-always-ram",
        dest="always_ram",
        action="store_false",
        help="Allow quantized vectors to live on disk.",
    )
    ingest_parser.set_defaults(func=ingest_documents)

    ingest_file_parser = subparsers.add_parser(
        "ingest-file",
        help="Chunk + embed local PDF/MD/TXT files and upsert them into Qdrant.",
    )
    ingest_file_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    ingest_file_parser.add_argument(
        "--category",
        default="uploaded",
        help="Category label written to each chunk's payload.",
    )
    ingest_file_parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags applied to every chunk.",
    )
    ingest_file_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help="Words per chunk (default: %(default)s).",
    )
    ingest_file_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Words of overlap between consecutive chunks (default: %(default)s).",
    )
    ingest_file_parser.add_argument(
        "--no-replace",
        dest="replace",
        action="store_false",
        help="Do NOT delete existing chunks for the same source before upserting.",
    )
    ingest_file_parser.add_argument(
        "--quantization",
        choices=list(QUANTIZATION_MODES),
        default="none",
        help=(
            "Quantization mode for the collection. Only applies when the "
            "collection is being created; ignored (with a warning) on "
            "existing collections — use 'quantize' to retrofit instead."
        ),
    )
    ingest_file_parser.add_argument(
        "--always-ram",
        dest="always_ram",
        action="store_true",
        default=True,
        help="Keep quantized vectors in RAM (default).",
    )
    ingest_file_parser.add_argument(
        "--no-always-ram",
        dest="always_ram",
        action="store_false",
        help="Allow quantized vectors to live on disk.",
    )
    ingest_file_parser.set_defaults(replace=True, func=ingest_files_command)
    ingest_file_parser.add_argument(
        "paths",
        nargs="+",
        help="One or more file paths (.pdf, .md, .markdown, .txt).",
    )

    query_parser = subparsers.add_parser(
        "query", help="Run semantic search against the collection."
    )
    query_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--limit", type=int, default=3)
    query_parser.add_argument(
        "--filter-category",
        dest="filter_category",
        default=None,
        help="Restrict results to this category value.",
    )
    query_parser.add_argument(
        "--filter-source",
        dest="filter_source",
        default=None,
        help="Restrict results to this source identifier (e.g. whitepaper.pdf).",
    )
    query_parser.add_argument(
        "--filter-source-type",
        dest="filter_source_type",
        default=None,
        choices=["pdf", "markdown", "text", "sample"],
        help="Restrict results to this source type.",
    )
    query_parser.add_argument(
        "--filter-tags",
        dest="filter_tags",
        default=None,
        help="Comma-separated tag values; all supplied tags must be present.",
    )
    query_parser.add_argument(
        "--rescore",
        dest="rescore",
        action="store_true",
        default=None,
        help=(
            "On quantized collections, re-score the top candidates with the "
            "original full-precision vectors (default Qdrant behavior). "
            "Pair with --oversampling for a recall demo."
        ),
    )
    query_parser.add_argument(
        "--no-rescore",
        dest="rescore",
        action="store_false",
        help="Skip full-precision re-scoring — fastest, lowest recall.",
    )
    query_parser.add_argument(
        "--oversampling",
        type=float,
        default=None,
        help=(
            "Multiplier for how many quantized candidates to fetch before "
            "re-scoring (e.g. 2.0). Higher = better recall, slower. Only "
            "meaningful on quantized collections."
        ),
    )
    query_parser.set_defaults(func=query_documents)

    traverse_parser = subparsers.add_parser(
        "traverse",
        help="Traverse points in the collection via Qdrant scroll pagination.",
    )
    traverse_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    traverse_parser.add_argument("--batch-size", type=int, default=4)
    traverse_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of points to print. 0 means no limit.",
    )
    traverse_parser.set_defaults(func=traverse_documents)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Run a tiny RAG chat endpoint backed by Qdrant retrieval + Ollama generation.",
    )
    serve_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    serve_parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    serve_parser.add_argument("--retrieval-limit", type=int, default=3)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=serve_chat_endpoint)

    memory_parser = subparsers.add_parser(
        "memory",
        help=(
            "Print collection stats and a side-by-side memory estimate "
            "across all quantization modes."
        ),
    )
    memory_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    memory_parser.set_defaults(func=memory_report)

    quantize_parser = subparsers.add_parser(
        "quantize",
        help="Apply quantization to an existing collection in-place.",
    )
    quantize_parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    quantize_parser.add_argument(
        "--mode",
        required=True,
        choices=[m for m in QUANTIZATION_MODES if m != "none"],
        help="Quantization mode to apply (scalar/binary/product).",
    )
    quantize_parser.add_argument(
        "--always-ram",
        dest="always_ram",
        action="store_true",
        default=True,
        help="Keep quantized vectors in RAM (default).",
    )
    quantize_parser.add_argument(
        "--no-always-ram",
        dest="always_ram",
        action="store_false",
        help="Allow quantized vectors to live on disk.",
    )
    quantize_parser.set_defaults(func=quantize_collection)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
