"""Tests for the chunkers."""

from typing import Callable, List

import pytest

from src.chunking import (
    _split_span,
    chunk_file,
    chunk_markdown,
    chunk_python,
    chunk_text,
)
from src.models import Chunk

Chunker = Callable[[str, str, int], List[Chunk]]

PY_SAMPLE = '''"""Module docstring."""

import os
import sys

X = 1


# A helper.
def helper(a, b):
    """Add."""
    return a + b


@decorator
@other
class Big:
    """A class."""

    attr = 1

    # leading comment
    @property
    def one(self):
        return 1

    def two(self):
        return "two" * 10

    async def three(self):
        return await something()


def last():
    pass
# trailing comment
'''

MD_SAMPLE = """Intro text before any heading.

# Title

Some paragraph.

## Section A

Paragraph A1 with some words.

Paragraph A2 with more words.

```bash
# not a heading
echo hi
```

## Section B

Tail.

~~~
## also not a heading
~~~
"""

TEXT_SAMPLE = "alpha beta gamma\n\ndelta epsilon\nzeta eta theta iota\n" * 5

INPUTS = [
    "",
    "   \n\n  ",
    "x",
    "a" * 500,
    "word " * 300,
    "no newline at all but several words in a single line " * 20,
    "héllo wörld ☃ 日本語のテキスト\n\n" * 30,
    "line\r\nwith\r\ncrlf\r\n" * 40,
    TEXT_SAMPLE,
    PY_SAMPLE,
    MD_SAMPLE,
    "def broken(:\n    pass\n" * 10,
    "# just a comment\n",
    "x = 1\x00\n",
]
SIZES = [1, 2, 7, 50, 200, 2000]
CHUNKERS: List[Chunker] = [
    chunk_python,
    chunk_markdown,
    chunk_text,
    chunk_file,
]


def check_invariants(text: str, chunks: List[Chunk], size: int) -> None:
    """Assert all the chunking invariants.

    Args:
        text: The original text.
        chunks: Chunks produced for it.
        size: The maximum chunk size used.
    """
    covered = [False] * len(text)
    previous_end = 0
    for chunk in chunks:
        first = chunk.first_character_index
        last = chunk.last_character_index
        assert chunk.text == text[first:last]
        assert 0 < len(chunk.text) <= size
        assert first >= previous_end
        previous_end = last
        for i in range(first, last):
            covered[i] = True
    for i, char in enumerate(text):
        if not char.isspace():
            assert covered[i], f"uncovered character at {i}"


@pytest.mark.parametrize("chunker", CHUNKERS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("text", INPUTS)
def test_invariants(chunker: Chunker, size: int, text: str) -> None:
    """Every chunker satisfies the invariants on every input and size."""
    check_invariants(text, chunker(text, "a/b.py", size), size)


@pytest.mark.parametrize("chunker", CHUNKERS)
@pytest.mark.parametrize("size", [0, -5])
def test_invalid_size(chunker: Chunker, size: int) -> None:
    """A size below 1 raises ValueError."""
    with pytest.raises(ValueError):
        chunker("abc", "a.py", size)


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_empty_text(chunker: Chunker) -> None:
    """Empty and whitespace-only texts give no chunks."""
    assert chunker("", "a.txt", 100) == []
    assert chunker(" \n\n\t", "a.txt", 100) == []


def test_split_span_prefers_blank_line() -> None:
    """Pieces end after a blank line when one is in the second half."""
    text = "aaaa\n\nbbbb\ncccc dddd"
    pieces = _split_span(text, 0, len(text), 12)
    assert pieces[0] == (0, 6)
    assert pieces[-1][1] == len(text)


def test_split_span_newline_then_space_then_hard() -> None:
    """Fallback order: newline, space, hard cut."""
    assert _split_span("aaaaaaa\nbbbbbbb", 0, 15, 10)[0] == (0, 8)
    assert _split_span("aaaaaaa bbbbbbb", 0, 15, 10)[0] == (0, 8)
    assert _split_span("a" * 25, 0, 25, 10) == [(0, 10), (10, 20), (20, 25)]


def test_split_span_uses_first_half_boundary_if_needed() -> None:
    """A boundary in the first half is used when the second has none."""
    pieces = _split_span("ab\ncdefghijklmnop", 0, 17, 10)
    assert pieces[0] == (0, 3)


def test_split_span_contiguous() -> None:
    """Pieces tile the span exactly."""
    text = TEXT_SAMPLE
    pieces = _split_span(text, 3, len(text) - 2, 17)
    assert pieces[0][0] == 3 and pieces[-1][1] == len(text) - 2
    for (_, a_end), (b_start, _) in zip(pieces, pieces[1:]):
        assert a_end == b_start
    assert all(0 < b - a <= 17 for a, b in pieces)


def test_python_small_file_single_chunk() -> None:
    """A file that fits in the limit yields one chunk."""
    chunks = chunk_python(PY_SAMPLE, "x.py", 10000)
    assert len(chunks) == 1
    assert chunks[0].kind == "python"
    assert chunks[0].text == PY_SAMPLE


def test_python_functions_separate() -> None:
    """Definitions are separate chunks and keep decorators and comments."""
    text = "import os\n\n\n" + "".join(
        f"# note {i}\n@deco\ndef f{i}():\n"
        + "".join(f"    x{j} = {j}\n" for j in range(20))
        + "\n\n"
        for i in range(4)
    )
    size = 400
    chunks = chunk_python(text, "x.py", size)
    assert len(chunks) >= 4
    for i in range(4):
        owners = [c for c in chunks if f"def f{i}():" in c.text]
        assert len(owners) == 1
        assert f"# note {i}\n@deco\ndef f{i}" in owners[0].text
    check_invariants(text, chunks, size)


def test_python_class_split_at_methods() -> None:
    """An oversized class is cut between methods, not inside them."""
    methods = "".join(
        f"    def m{i}(self):\n        return {i}\n\n" for i in range(30)
    )
    text = "class A:\n    '''Doc.'''\n\n" + methods
    chunks = chunk_python(text, "x.py", 300)
    assert len(chunks) > 3
    assert chunks[0].text.startswith("class A:")
    for chunk in chunks[1:]:
        assert chunk.text.lstrip().startswith("def m")
    check_invariants(text, chunks, 300)


def test_python_class_header_comment_goes_with_method() -> None:
    """A comment and decorator right above a method stay with it."""
    chunks = chunk_python(PY_SAMPLE, "x.py", 150)
    owner = [c for c in chunks if "def one" in c.text]
    assert len(owner) == 1
    assert "# leading comment\n    @property" in owner[0].text
    check_invariants(PY_SAMPLE, chunks, 150)


def test_python_syntax_error_falls_back_to_text() -> None:
    """Unparseable source is chunked as text."""
    chunks = chunk_python("def broken(:\n  pass\n", "x.py", 100)
    assert [c.kind for c in chunks] == ["text"]


def test_python_tiny_chunks_merged() -> None:
    """Small adjacent segments are merged into one chunk."""
    text = "import a\nimport b\n\ndef f():\n    pass\n"
    chunks = chunk_python(text, "x.py", 80)
    assert len(chunks) == 1


def test_python_big_function_is_split() -> None:
    """A function larger than the limit is split with the limit held."""
    text = "def f():\n" + "    x = 1\n" * 100
    chunks = chunk_python(text, "x.py", 100)
    assert len(chunks) > 5
    check_invariants(text, chunks, 100)


def test_markdown_sections_and_fences() -> None:
    """Headings split sections, fenced code lines do not."""
    text = (
        "# One\n\n" + "para one\n" * 40 + "\n## Two\n\n"
        + "```\n# fake\n```\n" + "para two\n" * 40
    )
    chunks = chunk_markdown(text, "d.md", 400)
    assert all(c.kind == "markdown" for c in chunks)
    starts = [c.text.startswith("## Two") for c in chunks]
    assert any(starts)
    for chunk in chunks:
        assert not chunk.text.startswith("# fake")
    check_invariants(text, chunks, 400)


def test_markdown_merges_short_sections() -> None:
    """Short sections are merged with the following one."""
    text = "# A\n\nshort\n\n## B\n\nalso short\n\n## C\n\n" + "z" * 300
    chunks = chunk_markdown(text, "d.md", 2000)
    assert len(chunks) == 1
    big = chunk_markdown(text, "d.md", 320)
    assert len(big) > 1
    check_invariants(text, big, 320)


def test_markdown_content_before_first_heading() -> None:
    """Text before the first heading is kept."""
    chunks = chunk_markdown("Preamble\n\n# H\n\n" + "b" * 300, "d.md", 310)
    assert chunks[0].text.startswith("Preamble")
    assert len(chunks) == 2


def test_text_kind() -> None:
    """Plain text chunks have kind text."""
    chunks = chunk_text("hello world " * 100, "a.txt", 100)
    assert chunks and all(c.kind == "text" for c in chunks)


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("a/b.py", "python"),
        ("a/b.md", "markdown"),
        ("a/b.mdx", "markdown"),
        ("a/b.rst", "markdown"),
        ("a/b.cu", "text"),
        ("a/Makefile", "text"),
    ],
)
def test_chunk_file_dispatch(path: str, kind: str) -> None:
    """The extension selects the chunker."""
    chunks = chunk_file("x = 1\n", path, 100)
    assert chunks[0].kind == kind
    assert chunks[0].file_path == path
