"""Command line interface of the RAG system (exposed through Fire)."""


class RagCLI:
    """RAG against the machine: index, search, answer and evaluate."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Index the raw corpus.

        Args:
            max_chunk_size: Maximum chunk size in characters.
            raw_dir: Directory holding the raw repository.
            processed_dir: Directory where the index is written.
        """
        print("index: not implemented yet")

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
        print("search: not implemented yet")

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
        print("search_dataset: not implemented yet")

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
        print("evaluate: not implemented yet")
