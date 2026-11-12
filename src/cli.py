"""Command line interface of the RAG system (exposed through Fire)."""

import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from src.evaluation import recall_table
from src.index import DEFAULT_EXCLUDE_DIRS, build_index
from src.models import (
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)
from src.retriever import IndexNotFoundError, Retriever


def _new_id() -> str:
    """Create a fresh question id.

    Returns:
        A UUID string.
    """
    return str(uuid.uuid4())


def _error(message: str) -> None:
    """Print an error message to stderr.

    Args:
        message: The message.
    """
    print(f"Error: {message}", file=sys.stderr)


def _is_int(value: Any) -> bool:
    """Tell whether a value is a real integer (not a bool).

    Args:
        value: Any value.

    Returns:
        True for ints that are not bools.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _check_k(k: Any) -> Optional[int]:
    """Validate k, printing a message when it is unusable.

    Args:
        k: The raw value received from the command line.

    Returns:
        The integer k, or None if invalid (error already printed).
    """
    if not _is_int(k):
        _error(f"k must be an integer, got {k!r}")
        return None
    if k <= 0:
        print("Warning: k must be positive, returning no sources.",
              file=sys.stderr)
    return int(k)


def _exclude_text(value: Any) -> str:
    """Normalize the ``exclude_dirs`` option (Fire may parse lists).

    Args:
        value: Raw value from the command line.

    Returns:
        A comma separated string.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _load_retriever(processed_dir: str) -> Optional[Retriever]:
    """Load the retriever, printing a message on failure.

    Args:
        processed_dir: Directory holding the index.

    Returns:
        The retriever or None.
    """
    try:
        return Retriever.load(processed_dir)
    except IndexNotFoundError as exc:
        _error(f"Index not found, run `index` first. {exc}")
        return None


def _load_model(path: str, model: Any) -> Any:
    """Load and validate a JSON file into a pydantic model.

    Args:
        path: File path.
        model: The pydantic model class.

    Returns:
        The validated instance, or None (error printed).
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return model.model_validate_json(handle.read())
    except FileNotFoundError:
        _error(f"file not found: {path}")
    except ValidationError as exc:
        _error(f"invalid file {path}: {exc}")
    except (OSError, UnicodeDecodeError) as exc:
        _error(f"cannot read {path}: {exc}")
    return None


class RagCLI:
    """RAG against the machine: index, search, answer and evaluate."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        exclude_dirs: str = DEFAULT_EXCLUDE_DIRS,
    ) -> None:
        """Index the raw corpus.

        Args:
            max_chunk_size: Maximum chunk size in characters.
            raw_dir: Directory holding the raw repository.
            processed_dir: Directory where the index is written.
            exclude_dirs: Comma separated directory names to skip
                (empty string excludes nothing).
        """
        if not _is_int(max_chunk_size) or max_chunk_size < 1:
            _error("max_chunk_size must be an integer >= 1")
            return
        if not Path(str(raw_dir)).is_dir():
            _error(f"raw_dir does not exist: {raw_dir}")
            return
        try:
            build_index(
                str(raw_dir),
                str(processed_dir),
                max_chunk_size,
                _exclude_text(exclude_dirs),
            )
        except Exception as exc:
            _error(f"indexing failed: {exc}")

    def search(
        self,
        query: str = "",
        k: int = 10,
        processed_dir: str = "data/processed",
    ) -> None:
        """Retrieve the top-k chunks for a single query.

        Args:
            query: The question to search for.
            k: Number of chunks to retrieve.
            processed_dir: Directory holding the index.
        """
        query = str(query)
        k_value = _check_k(k)
        if k_value is None:
            return
        if not query.strip():
            _error("empty query")
            return
        retriever = _load_retriever(str(processed_dir))
        if retriever is None:
            return
        chunks = retriever.search(query, k_value)
        result = MinimalSearchResults(
            question_id=_new_id(),
            question=query,
            retrieved_sources=[c.to_source() for c in chunks],
        )
        print(result.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str = (
            "data/datasets/UnansweredQuestions/dataset_docs_public.json"
        ),
        k: int = 10,
        save_directory: str = "data/output/search_results/"
        "UnansweredQuestions",
        processed_dir: str = "data/processed",
    ) -> None:
        """Retrieve the top-k chunks for every question of a dataset.

        Args:
            dataset_path: Path of the dataset JSON file.
            k: Number of chunks to retrieve per question.
            save_directory: Directory where the results are saved.
            processed_dir: Directory holding the index.
        """
        k_value = _check_k(k)
        if k_value is None:
            return
        dataset = _load_model(str(dataset_path), RagDataset)
        if dataset is None:
            return
        retriever = _load_retriever(str(processed_dir))
        if retriever is None:
            return
        questions = dataset.rag_questions
        found = retriever.search_many(
            [q.question for q in questions], k_value
        )
        results = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=[c.to_source() for c in chunks],
                )
                for q, chunks in zip(questions, found)
            ],
            k=k_value,
        )
        try:
            out_dir = Path(str(save_directory))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / Path(str(dataset_path)).name
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(results.model_dump_json(indent=2))
        except OSError as exc:
            _error(f"cannot save results: {exc}")
            return
        print(f"Saved student_search_results to {out_path}")

    def answer(
        self,
        query: str = "",
        k: int = 10,
        processed_dir: str = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Retrieve chunks and answer a single query.

        Args:
            query: The question to answer.
            k: Number of chunks to retrieve.
            processed_dir: Directory holding the index.
            model_name: Name of the generation model.
        """
        print("answer: not implemented yet")

    def answer_dataset(
        self,
        student_search_results_path: str = (
            "data/output/search_results/UnansweredQuestions/"
            "dataset_docs_public.json"
        ),
        save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"
        ),
        model_name: str = "Qwen/Qwen3-0.6B",
        raw_dir_root: str = ".",
    ) -> None:
        """Answer every question of a search-results file.

        Args:
            student_search_results_path: Path of the search results JSON.
            save_directory: Directory where the answers are saved.
            model_name: Name of the generation model.
            raw_dir_root: Root against which source file paths resolve.
        """
        print("answer_dataset: not implemented yet")

    def evaluate(
        self,
        student_search_results_path: str = (
            "data/output/search_results/UnansweredQuestions/"
            "dataset_docs_public.json"
        ),
        dataset_path: str = (
            "data/datasets/AnsweredQuestions/dataset_docs_public.json"
        ),
        k: int = 10,
    ) -> None:
        """Compute recall@k of search results against a reference dataset.

        Args:
            student_search_results_path: Path of the search results JSON.
            dataset_path: Path of the answered reference dataset.
            k: Number of retrieved sources considered.
        """
        k_value = _check_k(k)
        if k_value is None:
            return
        results = _load_model(
            str(student_search_results_path), StudentSearchResults
        )
        if results is None:
            return
        dataset = _load_model(str(dataset_path), RagDataset)
        if dataset is None:
            return
        table = recall_table(results, dataset)
        print(" ".join(f"Recall@{n}: {v:.3f}" for n, v in table.items()))
