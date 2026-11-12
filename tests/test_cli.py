"""CLI edge-case tests: nothing may raise."""

from pathlib import Path

import pytest

from src.cli import RagCLI


def test_missing_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Searching without an index prints an error."""
    RagCLI().search("lora", 5, str(tmp_path / "none"))
    assert "Index not found" in capsys.readouterr().err


def test_index_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bad chunk sizes and missing raw dirs are reported."""
    cli = RagCLI()
    cli.index("abc", str(tmp_path), str(tmp_path / "p"))  # type: ignore
    cli.index(True, str(tmp_path), str(tmp_path / "p"))
    cli.index(0, str(tmp_path), str(tmp_path / "p"))
    cli.index(100, str(tmp_path / "missing"), str(tmp_path / "p"))
    assert capsys.readouterr().err.count("Error") == 4


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
    cli.search(12345, 3, idx)  # type: ignore
    cli.search("", 3, idx)
    cli.search("lora", 0, idx)
    cli.search("lora", "abc", idx)  # type: ignore
    cli.search("lora", True, idx)
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
    cli.search_dataset(str(tmp_path / "x.json"), 5, str(tmp_path), "idx")
    cli.search_dataset(str(bad), 5, str(tmp_path), "idx")
    cli.evaluate(str(bad), str(bad), 5)
    cli.evaluate(str(tmp_path / "x.json"), str(bad), 5)
    cli.evaluate(str(bad), str(bad), "abc")  # type: ignore
    assert capsys.readouterr().err.count("Error") == 5
