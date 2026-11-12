"""Tests for the pydantic data models."""

from typing import Any, Dict

from src.models import (
    AnsweredQuestion,
    Chunk,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)

SOURCE: Dict[str, Any] = {
    "file_path": "data/raw/vllm-0.10.1/docs/a.md",
    "first_character_index": 0,
    "last_character_index": 10,
}


def test_answered_dataset_parses_as_answered() -> None:
    """Answered entries with extra keys parse as AnsweredQuestion."""
    data = {
        "rag_questions": [
            {
                "question_id": "q1",
                "question": "What?",
                "sources": [SOURCE],
                "answer": "That.",
                "difficulty": "easy",
                "is_valid": True,
            }
        ]
    }
    ds = RagDataset.model_validate(data)
    assert isinstance(ds.rag_questions[0], AnsweredQuestion)
    assert ds.rag_questions[0].sources[0].last_character_index == 10


def test_unanswered_dataset_parses_as_unanswered() -> None:
    """Unanswered entries parse as UnansweredQuestion."""
    data = {
        "rag_questions": [
            {"question_id": "q2", "question": "Why?", "is_valid": True}
        ]
    }
    ds = RagDataset.model_validate(data)
    q = ds.rag_questions[0]
    assert isinstance(q, UnansweredQuestion)
    assert not isinstance(q, AnsweredQuestion)


def test_question_id_defaults_to_uuid() -> None:
    """A missing question_id is generated."""
    assert len(UnansweredQuestion(question="x").question_id) == 36


def test_student_search_results_round_trip() -> None:
    """StudentSearchResults survives a JSON round trip."""
    res = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="What?",
                retrieved_sources=[MinimalSource(**SOURCE)],
            )
        ],
        k=5,
    )
    again = StudentSearchResults.model_validate_json(res.model_dump_json())
    assert again == res


def test_chunk_to_source() -> None:
    """Chunk.to_source keeps the location."""
    chunk = Chunk(kind="text", text="hello", **SOURCE)
    assert chunk.to_source() == MinimalSource(**SOURCE)
