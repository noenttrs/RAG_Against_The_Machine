"""Load a persisted BM25 index and retrieve chunks."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from src.models import Chunk
from src.tokenizer import tokenize


class IndexNotFoundError(Exception):
    """Raised when the index is missing or corrupt."""


class Retriever:
    """BM25 retriever over a persisted index."""

    def __init__(
        self,
        matrix: sp.csc_matrix,
        vocab: Dict[str, int],
        chunks: List[Chunk],
    ) -> None:
        """Create a retriever.

        Args:
            matrix: BM25 weights (chunks x terms), column-sliceable.
            vocab: Term to column id.
            chunks: Chunks aligned with the matrix rows.
        """
        self.matrix = matrix
        self.vocab = vocab
        self.chunks = chunks

    @classmethod
    def load(cls, processed_dir: str) -> "Retriever":
        """Load the index from disk.

        Args:
            processed_dir: Directory holding the index files.

        Returns:
            The retriever.

        Raises:
            IndexNotFoundError: If files are missing or corrupt.
        """
        base = Path(processed_dir)
        try:
            matrix = sp.load_npz(base / "bm25_matrix.npz").tocsc()
            with open(base / "vocab.json", encoding="utf-8") as handle:
                vocab = json.load(handle)
            with open(base / "chunks.json", encoding="utf-8") as handle:
                chunks = [Chunk(**c) for c in json.load(handle)]
        except Exception as exc:
            raise IndexNotFoundError(
                f"Index not found or corrupt in '{processed_dir}' "
                f"({exc}); run `index` first"
            ) from exc
        if matrix.shape[0] != len(chunks):
            raise IndexNotFoundError(
                f"Index in '{processed_dir}' is inconsistent; "
                "run `index` first"
            )
        return cls(matrix, vocab, chunks)

    def search(self, query: str, k: int) -> List[Chunk]:
        """Return the top-k chunks for a query.

        Args:
            query: The query text.
            k: Number of chunks wanted.

        Returns:
            Chunks by decreasing score, zero-score chunks excluded.
        """
        if k <= 0 or not query.strip():
            return []
        counts = Counter(
            self.vocab[t] for t in tokenize(query, True) if t in self.vocab
        )
        if not counts:
            return []
        ids = np.fromiter(counts.keys(), dtype=np.int64)
        weights = np.fromiter(counts.values(), dtype=np.float32)
        scores = np.asarray(self.matrix[:, ids] @ weights).ravel()
        k = min(k, len(scores))
        if k < len(scores):
            top = np.argpartition(-scores, k - 1)[:k]
        else:
            top = np.arange(len(scores))
        top = top[np.argsort(-scores[top], kind="stable")]
        return [self.chunks[i] for i in top if scores[i] > 0]

    def search_many(
        self, queries: Sequence[str], k: int
    ) -> List[List[Chunk]]:
        """Search several queries with a progress bar.

        Args:
            queries: The queries.
            k: Number of chunks per query.

        Returns:
            One result list per query.
        """
        return [
            self.search(q, k)
            for q in tqdm(queries, desc="Searching", unit="question")
        ]
