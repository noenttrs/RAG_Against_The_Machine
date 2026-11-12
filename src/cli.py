"""Command line interface of the RAG system (exposed through Fire)."""

import functools
import sys
import uuid
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
)

from pydantic import ValidationError
from tqdm import tqdm

from src.evaluation import recall_table
from src.generator import load_source_text
from src.index import DEFAULT_EXCLUDE_DIRS, build_index
from src.models import (
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retriever import IndexNotFoundError, Retriever


def _new_id() -> str:
    """Create a fresh question id.

    Returns:
        A UUID string.
    """
    return str(uuid.uuid4())


_F = TypeVar("_F", bound=Callable[..., None])


def _safe(func: _F) -> _F:
    """Turn any unexpected exception of a command into an error message.

    Args:
        func: A CLI command.

    Returns:
        The wrapped command, which never raises.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except Exception as exc:
            _error(f"{type(exc).__name__}: {exc}")

    return wrapper  # type: ignore[return-value]


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


class _Generator(Protocol):
    """Anything able to answer a question from chunks."""

    def generate(self, question: str, contexts: List[Chunk]) -> str:
        """Answer a question."""


GeneratorFactory = Callable[[str, int], _Generator]


def _default_factory(model_name: str, max_new_tokens: int) -> _Generator:
    """Create the real answer generator (imports torch lazily).

    Args:
        model_name: Hugging Face model name.
        max_new_tokens: Maximum number of generated tokens.

    Returns:
        The generator.
    """
    from src.generator import AnswerGenerator

    return AnswerGenerator(
        model_name, max_new_tokens=max_new_tokens
    )


def _make_generator(
    factory: GeneratorFactory, model_name: str, max_new_tokens: Any
) -> Optional[_Generator]:
    """Build a generator, printing a clean error if loading fails.

    Args:
        factory: Callable building the generator.
        model_name: Model name.
        max_new_tokens: Maximum number of generated tokens.

    Returns:
        The generator or None (error printed).
    """
    if not _is_int(max_new_tokens) or max_new_tokens < 1:
        _error("max_new_tokens must be an integer >= 1")
        return None
    try:
        return factory(model_name, int(max_new_tokens))
    except Exception as exc:
        _error(f"cannot load model {model_name!r}: {exc}")
        return None


def _answer_dataset(
    results_path: str,
    save_directory: str,
    model_name: str,
    raw_dir_root: str,
    max_new_tokens: Any,
    factory: GeneratorFactory,
) -> None:
    """Answer all questions of a search results file and save them.

    Args:
        results_path: Path of the search results JSON.
        save_directory: Directory where the answers are saved.
        model_name: Model name.
        raw_dir_root: Root against which source paths resolve.
        max_new_tokens: Maximum number of generated tokens.
        factory: Callable building the generator.
    """
    results = _load_model(results_path, StudentSearchResults)
    if results is None:
        return
    generator = _make_generator(factory, model_name, max_new_tokens)
    if generator is None:
        return
    total = len(results.search_results)
    print(f"Loaded {total} questions from {results_path}")
    cache: Dict[str, str] = {}
    answered: List[MinimalAnswer] = []
    for item in tqdm(results.search_results, desc="Answering"):
        chunks: List[Chunk] = []
        for src in item.retrieved_sources:
            text = load_source_text(src, raw_dir_root, cache)
            if text is not None:
                chunks.append(
                    Chunk(
                        file_path=src.file_path,
                        first_character_index=src.first_character_index,
                        last_character_index=src.last_character_index,
                        kind="source",
                        text=text,
                    )
                )
        try:
            text_answer = generator.generate(item.question, chunks)
        except Exception as exc:
            print(
                f"Warning: generation failed for {item.question_id}: "
                f"{exc}",
                file=sys.stderr,
            )
            text_answer = ""
        answered.append(
            MinimalAnswer(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=item.retrieved_sources,
                answer=text_answer,
            )
        )
    output = StudentSearchResultsAndAnswer(
        search_results=answered, k=results.k
    )
    try:
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(results_path).name
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(output.model_dump_json(indent=2))
    except OSError as exc:
        _error(f"cannot save answers: {exc}")
        return
    print(f"Saved student_search_results_and_answer to {out_path}")


class RagCLI:
    """RAG against the machine: index, search, answer and evaluate."""

    @_safe
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

    @_safe
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

    @_safe
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

    @_safe
    def answer(
        self,
        query: str = "",
        k: int = 10,
        processed_dir: str = "data/processed",
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
    ) -> None:
        """Retrieve chunks and answer a single query.

        Args:
            query: The question to answer.
            k: Number of chunks to retrieve.
            processed_dir: Directory holding the index.
            model_name: Name of the generation model.
            max_new_tokens: Maximum number of generated tokens.
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
        generator = _make_generator(
            _default_factory, str(model_name), max_new_tokens
        )
        if generator is None:
            return
        chunks = retriever.search(query, k_value)
        result = MinimalAnswer(
            question_id=_new_id(),
            question=query,
            retrieved_sources=[c.to_source() for c in chunks],
            answer=generator.generate(query, chunks),
        )
        print(result.model_dump_json(indent=2))

    @_safe
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
        max_new_tokens: int = 256,
    ) -> None:
        """Answer every question of a search-results file.

        Args:
            student_search_results_path: Path of the search results JSON.
            save_directory: Directory where the answers are saved.
            model_name: Name of the generation model.
            raw_dir_root: Root against which source file paths resolve.
            max_new_tokens: Maximum number of generated tokens.
        """
        _answer_dataset(
            str(student_search_results_path),
            str(save_directory),
            str(model_name),
            str(raw_dir_root),
            max_new_tokens,
            _default_factory,
        )

    @_safe
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
        """Compute recall@j (j in 1, 3, 5, 10 up to k, and k) of results.

        Args:
            student_search_results_path: Path of the search results JSON.
            dataset_path: Path of the answered reference dataset.
            k: At most this many retrieved sources are considered.
        """
        k_value = _check_k(k)
        if k_value is None:
            return
        if k_value <= 0:
            _error("k must be positive for evaluation")
            return
        results = _load_model(
            str(student_search_results_path), StudentSearchResults
        )
        if results is None:
            return
        dataset = _load_model(str(dataset_path), RagDataset)
        if dataset is None:
            return
        table = recall_table(results, dataset, k_value)
        print(" ".join(f"Recall@{n}: {v:.3f}" for n, v in table.items()))
