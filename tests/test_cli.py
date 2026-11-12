"""CLI edge-case tests: nothing may raise."""

from pathlib import Path
from typing import Callable

import pytest

from src.cli import RagCLI


def _exits_1(call: Callable[[], None]) -> None:
    """Assert that a command reports an error and exits with status 1."""
    with pytest.raises(SystemExit) as info:
        call()
    assert info.value.code == 1


def test_missing_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Searching without an index prints an error."""
    _exits_1(lambda: RagCLI().search("lora", 5, str(tmp_path / "none")))
    assert "Index not found" in capsys.readouterr().err


def test_index_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bad chunk sizes and missing raw dirs are reported."""
    cli = RagCLI()
    p = str(tmp_path / "p")
    _exits_1(lambda: cli.index("abc", str(tmp_path), p))  # type: ignore
    _exits_1(lambda: cli.index(True, str(tmp_path), p))
    _exits_1(lambda: cli.index(0, str(tmp_path), p))
    _exits_1(lambda: cli.index(100, str(tmp_path / "missing"), p))
    assert capsys.readouterr().err.count("Error") == 4


def test_index_max_chunk_size_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Chunk sizes above the moulinette limit (2000) are rejected."""
    _exits_1(lambda: RagCLI().index(2001, str(tmp_path), str(tmp_path / "p")))
    assert "between 1 and 2000" in capsys.readouterr().err
    assert not (tmp_path / "p").exists()


def test_flow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Index, search, search_dataset and evaluate end to end."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "lora.md").write_text("# LoRA\nAdapters for fine tuning.\n")
    (raw / "a.py").write_text("def foo():\n    return 1\n")
    idx = str(tmp_path / "idx")
    cli = RagCLI()
    cli.index(2000, str(raw), idx)
    capsys.readouterr()
    cli.search("lora adapters", 3, idx)
    assert "lora.md" in capsys.readouterr().out
    cli.search(12345, 3, idx)  # type: ignore  # coerced to a query
    capsys.readouterr()
    _exits_1(lambda: cli.search("", 3, idx))
    cli.search("lora", 0, idx)  # warning only, exit status 0
    assert "Warning" in capsys.readouterr().err
    _exits_1(lambda: cli.search("lora", "abc", idx))  # type: ignore
    _exits_1(lambda: cli.search("lora", True, idx))
    capsys.readouterr()
    ds = tmp_path / "ds.json"
    ds.write_text('{"rag_questions": [{"question_id": "1", '
                  '"question": "lora adapters", "sources": [], '
                  '"answer": "x"}]}')
    out = tmp_path / "out"
    cli.search_dataset(str(ds), 5, str(out), idx)
    assert "Saved student_search_results" in capsys.readouterr().out
    cli.evaluate(str(out / "ds.json"), str(ds), 10)
    assert "Recall@1:" in capsys.readouterr().out


def test_bad_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing and malformed files never raise."""
    cli = RagCLI()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    x = str(tmp_path / "x.json")
    _exits_1(lambda: cli.search_dataset(x, 5, str(tmp_path), "idx"))
    _exits_1(lambda: cli.search_dataset(str(bad), 5, str(tmp_path), "idx"))
    _exits_1(lambda: cli.evaluate(str(bad), str(bad), 5))
    _exits_1(lambda: cli.evaluate(x, str(bad), 5))
    _exits_1(lambda: cli.evaluate(str(bad), str(bad), "abc"))  # type: ignore
    assert capsys.readouterr().err.count("Error") == 5


def test_evaluate_k_zero_single_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """evaluate --k 0 prints exactly one error and no warning."""
    f = str(tmp_path / "x.json")
    _exits_1(lambda: RagCLI().evaluate(f, f, 0))
    err = capsys.readouterr().err
    assert err.count("Error") == 1
    assert "Warning" not in err
