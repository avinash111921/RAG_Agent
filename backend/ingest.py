from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

EMBED_BATCH_SIZE = 64 
MAX_RETRIES = 4

@dataclass
class IngestStats:
    files_found: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_skipped: int = 0
    pages: int = 0
    chunks: int = 0

def discover_pdfs(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00","")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path, source: str, file_hash: str) -> list[Document]:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            log.warning("Skipping %s: password-protected.", source)
            return []

        pages: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page": page_number,
                            "file_hash": file_hash
                        },
                    )
                )
    except Exception as exc:
        log.warning("Skipping %s: could not read PDF (%s).", source, exc)
        return []

    if not pages:
        log.warning(
            "Skipping %s: no extractable text. It is probably a scanned/image PDF and needs OCR.",
            source,
        )
    return pages


def build_splitter() -> RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

def chunk_id(chunk: Document) -> str:
    m = chunk.metadata
    key = f"{m['source']}|{m['page']}|{m['start_index']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def add_in_batches(store: Chroma, chunks: list[Document]) -> None:
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        ids = [chunk_id(c) for c in batch]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                store.add_documents(batch, ids=ids)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise
                wait = 2**attempt
                log.warning(
                    "Embedding batch failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)


def indexed_files(store: Chroma) -> dict[str, str]:
    records = store.get(include=["metadatas"])
    return {
        m["source"]: m.get("file_hash", "")
        for m in records["metadatas"]
        if m and "source" in m
    }


def prune_missing(store: Chroma, indexed: set[str], current_sources: set[str]) -> int:
    stale = indexed - current_sources
    for source in stale:
        store.delete(where={"source": source})
        log.info("Pruned %s (file no longer in data/).", source)
    return len(stale)


def ingest(data_dir: Path, reset: bool = False) -> IngestStats:
    stats = IngestStats()
    pdfs = discover_pdfs(data_dir)
    stats.files_found = len(pdfs)
    if not pdfs:
        log.warning("No PDFs found in %s. Add some and re-run.", data_dir)
        return stats

    log.info(
        "Provider: %s | Collection: %s | Chunking: %d chars, %d overlap",
        config.LLM_PROVIDER, config.COLLECTION_NAME, config.CHUNK_SIZE, config.CHUNK_OVERLAP,
    )
    store = config.get_vectorstore()
    if reset:
        log.info("--reset: clearing collection '%s'.", config.COLLECTION_NAME)
        store.reset_collection()

    splitter = build_splitter()
    already_indexed = indexed_files(store)
    current_sources: set[str] = set()

    for path in pdfs:
        source = path.relative_to(data_dir).as_posix()
        current_sources.add(source)

        file_hash = file_sha256(path)
        if already_indexed.get(source) == file_hash:
            stats.files_unchanged += 1
            log.info("Unchanged %-38s (skipped, already indexed)", source)
            continue

        pages = load_pdf(path, source, file_hash)
        if not pages:
            stats.files_skipped += 1
            continue

        chunks = splitter.split_documents(pages)

        store.delete(where={"source": source})
        try:
            add_in_batches(store, chunks)
        except Exception as exc:
            log.error("Failed to index %s after %d attempts: %s", source, MAX_RETRIES, exc)
            stats.files_skipped += 1
            continue

        stats.files_indexed += 1
        stats.pages += len(pages)
        stats.chunks += len(chunks)
        log.info("Indexed %-40s %4d pages -> %5d chunks", source, len(pages), len(chunks))

    prune_missing(store, set(already_indexed), current_sources)

    total = len(store.get(include=[])["ids"])
    log.info(
        "Done. %d/%d files indexed, %d unchanged, %d skipped | %d pages, %d new chunks. "
        "Collection now holds %d chunks in %s",
        stats.files_indexed, stats.files_found, stats.files_unchanged, stats.files_skipped,
        stats.pages, stats.chunks, total, config.CHROMA_DIR,
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the local ChromaDB vector store.")
    parser.add_argument(
        "--data-dir", type=Path, default=config.DATA_DIR,
        help=f"Folder containing PDFs (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete every chunk in the collection before ingesting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()

    if not data_dir.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
        log.warning("Created empty folder %s. Put your PDFs there and re-run.", data_dir)
        return 1

    try:
        stats = ingest(data_dir, reset=args.reset)
    except (RuntimeError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    return 0 if stats.files_indexed + stats.files_unchanged > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
