"""Tests for index building and retrieval."""

from pathlib import Path

import pytest

from src.index import build_index
from src.retriever import IndexNotFoundError, Retriever


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Create a tiny corpus.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        The corpus root.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "lora.md").write_text("# LoRA\nAdapters for fine tuning.\n")
    (raw / "sched.py").write_text(
        "def schedule_requests():\n    return 'continuous batching'\n"
    )
    (raw / "notes.txt").write_text("Install with pip on linux.\n")
    (raw / "data.json").write_text('{"lora": 1}')
    return raw


def test_build_and_search(corpus: Path, tmp_path: Path) -> None:
    """The obvious file ranks first and paths use the given raw dir."""
    out = tmp_path / "idx"
    n = build_index(str(corpus), str(out), 2000)
    assert n == 3
    retriever = Retriever.load(str(out))
    top = retriever.search("continuous batching scheduler", 3)
    assert top[0].file_path.endswith("sched.py")
    assert retriever.search("lora adapters", 1)[0].file_path.endswith(
        "lora.md"
    )
    assert all("json" not in c.file_path for c in retriever.search("lora", 9))


def test_edge_queries(corpus: Path, tmp_path: Path) -> None:
    """Empty, nonsense and k<=0 queries give no results."""
    out = tmp_path / "idx"
    build_index(str(corpus), str(out), 2000)
    retriever = Retriever.load(str(out))
    assert retriever.search("", 5) == []
    assert retriever.search("   ", 5) == []
    assert retriever.search("zzzzqqqq", 5) == []
    assert retriever.search("lora", 0) == []
    assert retriever.search("lora", -1) == []
    assert len(retriever.search("lora", 1000)) <= 3


def test_missing_index(tmp_path: Path) -> None:
    """Loading a missing index raises the custom error."""
    with pytest.raises(IndexNotFoundError):
        Retriever.load(str(tmp_path / "nope"))
