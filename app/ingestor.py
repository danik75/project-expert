import os
from typing import Callable

from app.config import (
    SUPPORTED_EXTENSIONS, SKIP_DIRS, MAX_FILE_SIZE_BYTES,
    MAX_CHUNK_CHARACTERS, CHUNK_OVERLAP_CHARACTERS,
)


def resolve_repo(source: str) -> str:
    """Validate that the given filesystem path exists and is a directory."""
    source = source.strip()
    if not source:
        raise ValueError("No project path provided.")

    abs_path = os.path.abspath(os.path.expanduser(source))
    if not os.path.isdir(abs_path):
        raise ValueError(f"Path does not exist or is not a directory: {abs_path}")
    return abs_path


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def discover_files(repo_path: str) -> list[str]:
    """Walk the project directory and return paths to all text source files worth indexing."""
    result = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue

            full_path = os.path.join(dirpath, filename)

            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue

            if size > MAX_FILE_SIZE_BYTES:
                continue

            if _is_binary(full_path):
                continue

            result.append(full_path)

    return result


def read_file_safe(path: str) -> str | None:
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, OSError):
            continue
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def chunk_text(text: str, file_path: str, chunk_size: int = MAX_CHUNK_CHARACTERS,
               overlap: int = CHUNK_OVERLAP_CHARACTERS) -> list[dict]:
    chunks = []
    start = 0
    chunk_index = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos + 1

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append({
                "text": chunk_content,
                "source": file_path,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        start = end - overlap if end < length else length

    return chunks


def ingest_repo(
    source: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict], str]:
    repo_path = resolve_repo(source)
    files = discover_files(repo_path)

    if not files:
        raise ValueError("No supported source files found in the project directory.")

    all_chunks: list[dict] = []
    total = len(files)

    for i, file_path in enumerate(files):
        if progress_callback:
            rel = os.path.relpath(file_path, repo_path)
            progress_callback(i + 1, total, rel)

        content = read_file_safe(file_path)
        if content is None:
            continue

        rel_path = os.path.relpath(file_path, repo_path)
        chunks = chunk_text(content, rel_path)
        all_chunks.extend(chunks)

    return all_chunks, repo_path
