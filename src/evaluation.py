"""Recall@k evaluation of search results."""

from typing import Dict, List

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

IOU_THRESHOLD = 0.05
KS = [1, 3, 5, 10]


def iou_match(found: MinimalSource, truth: MinimalSource) -> bool:
    """Tell whether a retrieved source matches a ground-truth source.

    Args:
        found: Retrieved source.
        truth: Ground-truth source.

    Returns:
        True if same file and character-range IoU >= 0.05.
    """
    if found.file_path != truth.file_path:
        return False
    inter = min(found.last_character_index, truth.last_character_index) - max(
        found.first_character_index, truth.first_character_index
    )
    if inter <= 0:
        return False
    union = max(found.last_character_index, truth.last_character_index) - min(
        found.first_character_index, truth.first_character_index
    )
    return inter / union >= IOU_THRESHOLD


def recall_at_k(
    results: StudentSearchResults, dataset: RagDataset, k: int
) -> float:
    """Compute the mean recall@k.

    Args:
        results: Student search results.
        dataset: Reference dataset.
        k: Number of top retrieved sources considered.

    Returns:
        Mean over answered questions of the found-sources fraction.
    """
    by_id: Dict[str, List[MinimalSource]] = {
        r.question_id: r.retrieved_sources for r in results.search_results
    }
    scores: List[float] = []
    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion) or not question.sources:
            continue
        retrieved = by_id.get(question.question_id, [])[:k]
        hits = sum(
            any(iou_match(r, s) for r in retrieved)
            for s in question.sources
        )
        scores.append(hits / len(question.sources))
    return sum(scores) / len(scores) if scores else 0.0


def recall_table(
    results: StudentSearchResults,
    dataset: RagDataset,
    max_k: int = 10,
) -> Dict[int, float]:
    """Compute recall for k in 1, 3, 5, 10 (limited to ``max_k``).

    ``max_k`` itself is added when it is not one of the standard values.

    Args:
        results: Student search results.
        dataset: Reference dataset.
        max_k: Largest k reported.

    Returns:
        Mapping from k to recall.
    """
    ks = sorted({k for k in KS if k <= max_k} | {max_k})
    return {k: recall_at_k(results, dataset, k) for k in ks}
