"""Tests for index building and retrieval."""

import json
from pathlib import Path

import pytest

from src.chunking import chunk_markdown, markdown_heading_contexts
from src.index import build_index, list_files, parse_exclude_dirs
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


def test_exclude_dirs(tmp_path: Path) -> None:
    """Only top-level repo directories are excluded."""
    raw = tmp_path / "raw"
    for rel in ("repo/docs/a.md", "repo/tests/b.py", "repo/x/examples/c.py",
                "repo/tests.md", "tests/d.md"):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        (raw / rel).write_text("hello")
    names = {
        p.relative_to(raw).as_posix()
        for p in list_files(str(raw), parse_exclude_dirs("tests,examples"))
    }
    assert names == {
        "repo/docs/a.md", "repo/tests.md", "tests/d.md", "repo/x/examples/c.py"
    }
    assert len(list_files(str(raw), parse_exclude_dirs(""))) == 5


def test_heading_context_and_trim() -> None:
    """Parent headings are found; trimming keeps text slices exact."""
    text = "# Top\n\n## Sub\n\n" + "word " * 300 + "\n\n### Deep\n\n" + (
        "more " * 300
    )
    chunks = chunk_markdown(text, "f.md", 1000, trim=True)
    for c in chunks:
        assert text[c.first_character_index:c.last_character_index] == c.text
        assert c.text == c.text.strip()
    ctx = markdown_heading_contexts(text, chunks)
    assert ctx[0] == ""
    assert any("Top Sub" in c for c in ctx[1:])


def test_exclude_keeps_nested_package(tmp_path: Path) -> None:
    """repo/pkg/benchmarks is kept, repo/benchmarks is excluded."""
    raw = tmp_path / "raw"
    for rel in ("repo/pkg/benchmarks/x.py", "repo/benchmarks/x.py"):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        (raw / rel).write_text("x = 1")
    names = [
        p.relative_to(raw).as_posix()
        for p in list_files(str(raw), parse_exclude_dirs("benchmarks"))
    ]
    assert names == ["repo/pkg/benchmarks/x.py"]


def test_meta_records_exclude_dirs(corpus: Path, tmp_path: Path) -> None:
    """meta.json stores the exclude_dirs option."""
    out = tmp_path / "idx"
    build_index(str(corpus), str(out), 2000, "foo,bar")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["exclude_dirs"] == "foo,bar"


def test_corrupt_vocab(corpus: Path, tmp_path: Path) -> None:
    """A vocab id beyond the matrix width is rejected."""
    out = tmp_path / "idx"
    build_index(str(corpus), str(out), 2000)
    vocab = json.loads((out / "vocab.json").read_text())
    vocab["bogus"] = 10 ** 6
    (out / "vocab.json").write_text(json.dumps(vocab))
    with pytest.raises(IndexNotFoundError):
        Retriever.load(str(out))
