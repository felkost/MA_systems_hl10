"""Build the knowledge index from `data/`.

Run as a separate command: `python ingest.py`. The index is built once and
read many times, so retrieval never triggers embedding of `data/`.

A repeated run is free: chunk ids are derived from the file contents, so a
source file that is already indexed is skipped instead of embedded again.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ValidationError

import models
import paths
from config import Settings, load_settings

TEXT_SUFFIXES = frozenset({".txt", ".md"})
PDF_SUFFIX = ".pdf"
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {PDF_SUFFIX}


@dataclass(frozen=True)
class IngestStats:
    """What one ingestion run did.

    Attributes
    ----------
    files, pages, chunks : int
        Source files read, page-level documents loaded, chunks in the index.
    added : int
        Chunks embedded and written during this run. Zero means every source
        file was already indexed and unchanged.
    seconds : float
        Wall time of the run.
    """

    files: int
    pages: int
    chunks: int
    added: int
    seconds: float


def ingest(settings: Settings, embeddings: Embeddings | None = None) -> IngestStats:
    """Turn the documents in `data_dir` into a searchable index.

    Parameters
    ----------
    settings : Settings
        `data_dir`, `index_dir`, `collection_name`, `chunk_size`,
        `chunk_overlap` and the embedding fields are read here.
    embeddings : Embeddings, optional
        Embedding backend. Defaults to `models.build_embeddings(settings)`.
        Passing one explicitly allows ingestion to run without network
        access.

    Returns
    -------
    IngestStats
        Counts and wall time of the run.

    Raises
    ------
    FileNotFoundError
        If the data directory is missing or holds no supported document.
    """
    started = time.perf_counter()
    data_dir = paths.resolve(settings.data_dir)
    index_dir = paths.resolve(settings.index_dir)

    documents = _load_documents(data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    index_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or models.build_embeddings(settings),
        persist_directory=str(index_dir),
    )
    added = _synchronize(store, chunks)

    files = len({document.metadata["source"] for document in documents})
    _write_corpus(chunks, paths.corpus_path(index_dir))
    _write_manifest(
        settings,
        files=files,
        chunks=len(chunks),
        sources=source_hashes(data_dir),
        path=paths.manifest_path(index_dir),
    )

    return IngestStats(
        files=files,
        pages=len(documents),
        chunks=len(chunks),
        added=added,
        seconds=round(time.perf_counter() - started, 2),
    )


def source_hashes(data_dir: Path) -> dict[str, str]:
    """SHA-256 of each supported source file's raw bytes, keyed by filename.

    Read by `retriever.py`'s `_verify_manifest` to detect a `data/` that no
    longer matches what `manifest.json` recorded. Hashing bytes, not chunk
    text, is deliberate: chunk boundaries move when `chunk_size`/
    `chunk_overlap` change even though the file did not, and the file hash
    must not move with them.

    Returns
    -------
    dict
        Empty if `data_dir` does not exist -- "every source removed" is a
        meaningful, reportable state for a caller comparing against a
        manifest, not an error condition of this function.
    """
    if not data_dir.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(data_dir.iterdir())
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    }


def _load_documents(data_dir: Path) -> list[Document]:
    """Read every supported file as one document per page.

    Metadata is rebuilt instead of reused: pypdf returns a producer, a
    creation date and other values that differ between machines, and Chroma
    rejects the `None` values among them. `source` is the file name, not the
    full path, so a chunk id does not depend on the location of the checkout.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data directory not found: {data_dir}. Put the source documents "
            "there, or point DATA_DIR at the right place."
        )

    documents: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix == PDF_SUFFIX:
            documents.extend(_read_pdf(path))
        elif suffix in TEXT_SUFFIXES:
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name, "page": 0},
                )
            )

    if not documents:
        raise FileNotFoundError(f"no .pdf, .txt or .md files in {data_dir}")
    return documents


def _read_pdf(path: Path) -> list[Document]:
    reader = pypdf.PdfReader(str(path))
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": path.name, "page": number},
        )
        for number, page in enumerate(reader.pages)
    ]


def _chunk_id(chunk: Document) -> str:
    """Derive a stable id from the chunk's provenance and its text.

    Without explicit ids Chroma assigns a new UUID on every run and duplicates
    the whole corpus. Retrieval then returns the same passage several times,
    and nothing reports an error.

    The chunk text is part of the key, not only its position. An edit that
    does not move the offsets after it -- a corrected word, a replaced
    sentence -- leaves `source|page|start_index` unchanged, so an id built
    from the position alone would mark the new text as already indexed and the
    old text would stay in the index.
    """
    meta = chunk.metadata
    key = (
        f"{meta['source']}|{meta['page']}|{meta['start_index']}"
        f"|{chunk.page_content}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _synchronize(store: Chroma, chunks: list[Document]) -> int:
    """Bring the collection in line with `chunks`, file by file.

    Returns the number of chunks embedded. Chroma embeds everything passed to
    `add_documents`, with or without ids, so a repeated run is only free if
    whole unchanged files are skipped. A file whose chunk set changed is
    replaced completely: an edit moves every offset after it, and adding the
    new chunks without deleting the old ones would leave the previous text in
    the index, where it can still be retrieved.

    A source file removed from the input entirely is pruned by
    `_prune_removed_sources`: this loop only ever visits sources present in
    `chunks`, so a source that disappeared from `data_dir` would otherwise
    never be revisited, let alone deleted.
    """
    added = 0
    current_sources: set[str] = set()
    for source, source_chunks in _by_source(chunks).items():
        current_sources.add(source)
        wanted = [_chunk_id(chunk) for chunk in source_chunks]
        stored = store.get(where={"source": source}, include=[])["ids"]
        if set(stored) == set(wanted):
            continue
        if stored:
            store.delete(ids=stored)
        store.add_documents(source_chunks, ids=wanted)
        added += len(wanted)
    _prune_removed_sources(store, current_sources)
    return added


def _prune_removed_sources(store: Chroma, current_sources: set[str]) -> None:
    """Delete every chunk whose source file is no longer in `data_dir`.

    A deleted source must not go on answering `knowledge_search` forever just
    because nothing else in `_synchronize` looks for a source that vanished
    from the current `chunks` rather than merely changing.
    """
    stored_sources = {
        metadata["source"] for metadata in store.get(include=["metadatas"])["metadatas"]
    }
    for source in stored_sources - current_sources:
        stale_ids = store.get(where={"source": source}, include=[])["ids"]
        if stale_ids:
            store.delete(ids=stale_ids)


def _by_source(chunks: list[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.metadata["source"]].append(chunk)
    return grouped


def _write_corpus(chunks: list[Document], path: Path) -> None:
    payload = [
        {
            "id": _chunk_id(chunk),
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_manifest(
    settings: Settings,
    *,
    files: int,
    chunks: int,
    sources: dict[str, str],
    path: Path,
) -> None:
    """Record what the index was built with.

    `retriever.py` refuses an index whose manifest does not match the
    current settings or the current `data/`. A different embedding
    fingerprint or an edited source does not raise an error on its own; it
    returns wrong or stale results.
    """
    manifest: dict[str, Any] = {
        "collection_name": settings.collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_fingerprint": models.embedding_fingerprint(settings),
        "sources": sources,
        "files": files,
        "chunks": chunks,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    try:
        settings = load_settings()
    except ValidationError:
        print("Configuration error: check the provider keys and values in .env.")
        raise SystemExit(1)

    print(f"Reading {paths.resolve(settings.data_dir)}")

    try:
        stats = ingest(settings)
    except FileNotFoundError as error:
        print(f"Ingestion failed: {error}")
        raise SystemExit(1)

    print(
        f"Indexed {stats.chunks} chunks from {stats.pages} pages "
        f"of {stats.files} files in {stats.seconds}s "
        f"({stats.added} embedded this run)."
    )
    print(f"Index: {paths.resolve(settings.index_dir)}")


if __name__ == "__main__":
    main()
