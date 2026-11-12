"""Build and persist a BM25 index of the raw corpus."""

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from src.chunking import chunk_file, markdown_heading_contexts
from src.models import Chunk
from src.tokenizer import tokenize

K1 = 1.2
B = 0.75
PATH_REPEAT = 2
DEFAULT_EXCLUDE_DIRS = (
    "tests,benchmarks,examples,csrc,.buildkite,.github,tools"
)
BIGRAM_ALL = False
TRIM_CHUNKS = True
HEADING_CONTEXT = True
HEADING_REPEAT = 1
MAX_FILE_SIZE = 1_000_000
EXTENSIONS = frozenset(
    [".py", ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".sh",
     ".jinja", ".cfg"]
)
_PATH_SPLIT_RE = re.compile(r"[/._\-]+")


def parse_exclude_dirs(exclude_dirs: str) -> FrozenSet[str]:
    """Parse a comma separated list of directory names.

    Args:
        exclude_dirs: Comma separated names (empty means none).

    Returns:
        The set of names.
    """
    return frozenset(n.strip() for n in exclude_dirs.split(",") if n.strip())


def list_files(
    raw_dir: str, exclude: FrozenSet[str] = frozenset()
) -> List[Path]:
    """List indexable files below a directory, sorted.

    Args:
        raw_dir: Root directory of the corpus.
        exclude: Directory names skipped on any path component below
            the repository root (the first component under ``raw_dir``).

    Returns:
        Sorted list of file paths (as joined with ``raw_dir``).
    """
    root = Path(raw_dir)
    files = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in EXTENSIONS or not path.is_file():
            continue
        if exclude & set(path.relative_to(root).parts[1:-1]):
            continue
        if path.stat().st_size > MAX_FILE_SIZE:
            continue
        files.append(path)
    return files


def indexed_text(chunk: Chunk, context: str = "") -> str:
    """Build the text that is indexed for a chunk.

    Args:
        chunk: The chunk.
        context: Parent headings text (markdown), repeated
            ``HEADING_REPEAT`` times.

    Returns:
        Path words (repeated) followed by the chunk text.
    """
    words = " ".join(w for w in _PATH_SPLIT_RE.split(chunk.file_path) if w)
    head = " ".join([words] * PATH_REPEAT)
    if context:
        head += "\n" + " ".join([context] * HEADING_REPEAT)
    return head + "\n" + chunk.text


def build_matrix(
    docs: List[List[str]],
) -> Tuple[sp.csr_matrix, Dict[str, int]]:
    """Build the BM25 weight matrix.

    Args:
        docs: Tokenized chunks.

    Returns:
        The CSR matrix (chunks x terms) and the vocabulary.
    """
    vocab: Dict[str, int] = {}
    indptr = [0]
    indices: List[int] = []
    counts: List[int] = []
    lengths = np.zeros(len(docs), dtype=np.float64)
    for i, tokens in enumerate(docs):
        lengths[i] = len(tokens)
        for term, freq in Counter(tokens).items():
            indices.append(vocab.setdefault(term, len(vocab)))
            counts.append(freq)
        indptr.append(len(indices))
    n = len(docs)
    tf = np.asarray(counts, dtype=np.float64)
    cols = np.asarray(indices, dtype=np.int64)
    ptr = np.asarray(indptr, dtype=np.int64)
    df = np.bincount(cols, minlength=len(vocab)).astype(np.float64)
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
    avgdl = lengths.mean() if n and lengths.mean() > 0 else 1.0
    row_len = np.repeat(lengths, np.diff(ptr))
    norm = K1 * (1.0 - B + B * row_len / avgdl)
    data = idf[cols] * tf * (K1 + 1.0) / (tf + norm)
    matrix = sp.csr_matrix(
        (data, cols, ptr), shape=(n, len(vocab)), dtype=np.float32
    )
    return matrix, vocab


def build_index(
    raw_dir: str,
    processed_dir: str,
    max_chunk_size: int,
    exclude_dirs: str = DEFAULT_EXCLUDE_DIRS,
) -> int:
    """Chunk the corpus, build the BM25 index and save it.

    Args:
        raw_dir: Root directory of the corpus.
        processed_dir: Directory where the index files are written.
        max_chunk_size: Maximum chunk size in characters.
        exclude_dirs: Comma separated directory names to skip.

    Returns:
        The number of chunks indexed.
    """
    start = time.time()
    chunks: List[Chunk] = []
    docs: List[List[str]] = []
    files = list_files(raw_dir, parse_exclude_dirs(exclude_dirs))
    for path in tqdm(files, desc="Indexing", unit="file"):
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        file_path = path.as_posix()
        if file_path.startswith("./"):
            file_path = file_path[2:]
        file_chunks = chunk_file(text, file_path, max_chunk_size, TRIM_CHUNKS)
        contexts = [""] * len(file_chunks)
        if HEADING_CONTEXT and path.suffix in (".md", ".rst"):
            contexts = markdown_heading_contexts(text, file_chunks)
        for chunk, context in zip(file_chunks, contexts):
            chunks.append(chunk)
            docs.append(
                tokenize(
                    indexed_text(chunk, context),
                    BIGRAM_ALL or path.suffix != ".py",
                )
            )
    matrix, vocab = build_matrix(docs)
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "bm25_matrix.npz", matrix)
    with open(out / "vocab.json", "w", encoding="utf-8") as handle:
        json.dump(vocab, handle)
    with open(out / "chunks.json", "w", encoding="utf-8") as handle:
        json.dump([c.model_dump() for c in chunks], handle)
    meta = {
        "max_chunk_size": max_chunk_size,
        "k1": K1,
        "b": B,
        "n_chunks": len(chunks),
        "raw_dir": raw_dir,
        "exclude_dirs": exclude_dirs,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    print(f"Ingestion complete! Indices saved under {processed_dir}/")
    print(f"Elapsed: {time.time() - start:.1f}s ({len(chunks)} chunks)")
    return len(chunks)


__all__ = ["build_index", "build_matrix"]
