"""Tests for recall evaluation."""

from src.evaluation import iou_match, recall_at_k
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def src(path: str, a: int, b: int) -> MinimalSource:
    """Build a source.

    Args:
        path: File path.
        a: First index.
        b: Last index.

    Returns:
        The source.
    """
    return MinimalSource(
        file_path=path, first_character_index=a, last_character_index=b
    )


def test_iou() -> None:
    """IoU threshold and file matching."""
    assert not iou_match(src("f", 0, 100), src("f", 95, 195))
    assert iou_match(src("f", 0, 100), src("f", 90, 190))
    assert not iou_match(src("f", 0, 100), src("g", 0, 100))
    assert not iou_match(src("f", 0, 10), src("f", 10, 20))


def test_recall() -> None:
    """Recall averages per-question fractions; missing counts as 0."""
    dataset = RagDataset(rag_questions=[
        AnsweredQuestion(
            question_id="1", question="q", answer="a",
            sources=[src("f", 0, 100), src("g", 0, 100)],
        ),
        AnsweredQuestion(
            question_id="2", question="q", answer="a",
            sources=[src("h", 0, 100)],
        ),
    ])
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="1", question="q",
                retrieved_sources=[src("x", 0, 5), src("f", 0, 100)],
            )
        ],
        k=2,
    )
    assert recall_at_k(results, dataset, 1) == 0.0
    assert recall_at_k(results, dataset, 2) == 0.25
