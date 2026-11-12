"""Tests for answer generation (no real model)."""

import json
from pathlib import Path
from typing import List

import pytest

from src.cli import RagCLI, _answer_dataset
from src.generator import (
    NO_CONTEXT,
    AnswerGenerator,
    build_messages,
    load_source_text,
    strip_thinking,
)
from src.models import (
    Chunk,
    MinimalSource,
    StudentSearchResultsAndAnswer,
)


def _chunk(text: str = "hello") -> Chunk:
    """Make a chunk."""
    return Chunk(
        file_path="docs/a.md",
        first_character_index=0,
        last_character_index=len(text),
        kind="markdown",
        text=text,
    )


def test_build_messages() -> None:
    """Context blocks are numbered and the question comes last."""
    msgs = build_messages("Why?", [_chunk("abc"), _chunk("def")])
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "[1] docs/a.md (chars 0-3)\nabc" in user
    assert "[2] docs/a.md" in user
    assert user.endswith("Question: Why?")


def test_build_messages_empty() -> None:
    """Empty contexts produce the placeholder block."""
    assert NO_CONTEXT in build_messages("Q", [])[1]["content"]


def test_strip_thinking() -> None:
    """Think sections are removed."""
    assert strip_thinking("<think>\nx\n</think>\n\nAnswer ") == "Answer"
    assert strip_thinking("x</think> Answer") == "Answer"
    assert strip_thinking("plain") == "plain"


def test_load_source_text(tmp_path: Path) -> None:
    """Slices are read from disk, cached, and missing files give None."""
    (tmp_path / "f.txt").write_text("0123456789")
    src = MinimalSource(
        file_path="f.txt", first_character_index=2, last_character_index=5
    )
    cache: dict[str, str] = {}
    assert load_source_text(src, str(tmp_path), cache) == "234"
    assert "f.txt" in cache
    gone = MinimalSource(
        file_path="no.txt", first_character_index=0, last_character_index=1
    )
    assert load_source_text(gone, str(tmp_path)) is None


class _FakeGen:
    """Fake generator."""

    def generate(self, question: str, contexts: List[Chunk]) -> str:
        """Echo the number of contexts."""
        if "boom" in question:
            raise RuntimeError("boom")
        return f"{len(contexts)} contexts"


def _results_file(tmp_path: Path) -> Path:
    """Write a search results file and a source file."""
    (tmp_path / "a.md").write_text("0123456789")
    src = {
        "file_path": "a.md",
        "first_character_index": 0,
        "last_character_index": 4,
    }
    missing = dict(src, file_path="gone.md")
    data = {
        "k": 7,
        "search_results": [
            {"question_id": "1", "question": "ok",
             "retrieved_sources": [src, missing]},
            {"question_id": "2", "question": "boom",
             "retrieved_sources": []},
        ],
    }
    path = tmp_path / "res.json"
    path.write_text(json.dumps(data))
    return path


def test_answer_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Answers are written, k is copied, failures give empty answers."""
    path = _results_file(tmp_path)
    out = tmp_path / "out"
    _answer_dataset(
        str(path), str(out), "m", str(tmp_path), 16,
        lambda name, n: _FakeGen(),
    )
    captured = capsys.readouterr()
    assert "Loaded 2 questions" in captured.out
    assert "Saved student_search_results_and_answer" in captured.out
    result = StudentSearchResultsAndAnswer.model_validate_json(
        (out / "res.json").read_text()
    )
    assert result.k == 7
    assert [a.answer for a in result.search_results] == ["1 contexts", ""]
    assert result.search_results[0].retrieved_sources[0].file_path == "a.md"


def test_model_load_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model that cannot load gives a clean error."""
    path = _results_file(tmp_path)

    def factory(name: str, n: int) -> _FakeGen:
        raise OSError("offline")

    _answer_dataset(
        str(path), str(tmp_path / "o"), "m", str(tmp_path), 16, factory
    )
    err = capsys.readouterr().err
    assert "cannot load model" in err
    assert not (tmp_path / "o").exists()


def test_cli_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI answer commands never raise on bad input."""
    cli = RagCLI()
    cases = [
        (lambda: cli.answer("", 3, str(tmp_path)), "empty query"),
        (lambda: cli.answer("q", "x", str(tmp_path)),  # type: ignore
         "k must be an integer"),
        (lambda: cli.answer("q", 3, str(tmp_path / "none")),
         "Index not found"),
        (lambda: cli.answer_dataset(str(tmp_path / "missing.json")),
         "file not found"),
    ]
    for call, message in cases:
        call()
        assert message in capsys.readouterr().err


def test_generate_with_fakes() -> None:
    """generate slices prompt tokens, strips thinking, disables thinking."""
    import torch

    calls: List[dict] = []  # type: ignore[type-arg]

    class Tok:
        """Fake tokenizer."""

        pad_token_id = 0

        def apply_chat_template(self, messages: List[dict],  # type: ignore
                                **kwargs: object) -> str:
            """Record the call."""
            calls.append(kwargs)
            return "PROMPT"

        def encode(self, text: str, add_special_tokens: bool = False
                   ) -> List[str]:
            """Split."""
            return text.split()

        def __call__(  # type: ignore[type-arg]
            self, text: str, return_tensors: str
        ) -> dict:
            """Return 3 prompt tokens."""
            return {"input_ids": torch.tensor([[5, 6, 7]])}

        def decode(self, ids: torch.Tensor,
                   skip_special_tokens: bool = True) -> str:
            """Return text only if the ids are the new ones."""
            assert ids.tolist() == [8, 9]
            return "<think>hm</think> Answer "

    class Model:
        """Fake model."""

        device = torch.device("cpu")

        def eval(self) -> None:
            """No-op."""

        def generate(self, **kwargs: object) -> torch.Tensor:
            """Append two tokens to the prompt."""
            assert kwargs["do_sample"] is False
            return torch.tensor([[5, 6, 7, 8, 9]])

    gen = AnswerGenerator(model=Model(), tokenizer=Tok())
    assert gen.generate("Q?", [_chunk("abc")]) == "Answer"
    assert calls[0]["enable_thinking"] is False
    assert calls[0]["add_generation_prompt"] is True


def test_fit_contexts_truncates() -> None:
    """Contexts are cut to the token budget (whitespace tokenizer)."""

    class Tok:
        """Whitespace tokenizer."""

        def encode(self, text: str, add_special_tokens: bool = False
                   ) -> List[str]:
            """Split."""
            return text.split()

        def decode(self, ids: List[str]) -> str:
            """Join."""
            return " ".join(ids)

    gen = AnswerGenerator.__new__(AnswerGenerator)
    gen.tokenizer = Tok()
    gen.max_context_tokens = 20
    big = _chunk(" ".join(f"w{i}" for i in range(100)))
    kept = gen.fit_contexts([_chunk("a b c"), big, _chunk("z")])
    assert len(kept) == 2
    assert kept[0].text == "a b c"
    assert 0 < len(kept[1].text.split()) < 100
