"""Split source files into chunks bounded by a maximum character size.

Every chunker returns sorted, non-overlapping chunks, each at most
``max_chunk_size`` characters long, which together cover every
non-whitespace character of the file (whitespace-only chunks are dropped).
"""

import ast
import re
from typing import List, Sequence, Tuple

from src.models import Chunk

Span = Tuple[int, int]

_NEWLINE_RE = re.compile(r"\r\n|\r|\n")
_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_MIN_SECTION_SIZE = 400
_PY_MERGE_FRACTION = 0.5
_MARKDOWN_SUFFIXES = (".md", ".mdx", ".rst")


def _check_size(max_chunk_size: int) -> None:
    """Validate the maximum chunk size.

    Args:
        max_chunk_size: Requested maximum chunk size in characters.

    Raises:
        ValueError: If the size is smaller than 1.
    """
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")


def _find_cut(text: str, pos: int, limit: int, min_cut: int) -> int:
    """Find the best place to cut ``text[pos:limit]``.

    Boundaries are tried by priority (blank line, newline, space), first
    keeping only cuts at or after ``min_cut``, then anywhere in the window.

    Args:
        text: The full text.
        pos: Start of the window (inclusive).
        limit: End of the window (exclusive); the largest allowed cut.
        min_cut: Smallest cut accepted in the first, preferred pass.

    Returns:
        The exclusive end offset of the piece starting at ``pos``.
    """
    for lowest in (max(min_cut, pos + 1), pos + 1):
        for sep in ("\n\n", "\n", " "):
            idx = text.rfind(sep, pos, limit)
            if idx != -1 and idx + len(sep) >= lowest:
                return idx + len(sep)
    return limit


def _split_span(
    text: str, start: int, end: int, max_size: int
) -> List[Span]:
    """Split ``[start, end)`` into contiguous pieces of at most max_size.

    Cuts are made right after a blank line, else after a newline, else
    after a space, else hard. Each piece is the largest one that ends on
    the best boundary found in the second half of the window; when the
    second half has no boundary, any boundary in the window is used.

    Args:
        text: The full text.
        start: Start offset of the span (inclusive).
        end: End offset of the span (exclusive).
        max_size: Maximum piece size in characters (at least 1).

    Returns:
        Contiguous ``(start, end)`` pieces covering the whole span.
    """
    pieces: List[Span] = []
    pos = start
    while pos < end:
        if end - pos <= max_size:
            pieces.append((pos, end))
            break
        limit = pos + max_size
        cut = _find_cut(text, pos, limit, pos + max_size // 2)
        pieces.append((pos, cut))
        pos = cut
    return pieces


def _pack(spans: Sequence[Span], limit: int) -> List[Span]:
    """Greedily merge consecutive contiguous spans up to ``limit`` chars.

    A span larger than ``limit`` on its own is kept unchanged.

    Args:
        spans: Contiguous ``(start, end)`` spans in order.
        limit: Maximum combined size of merged spans.

    Returns:
        The merged spans.
    """
    packed: List[Span] = []
    for start, end in spans:
        if packed and end - packed[-1][0] <= limit:
            packed[-1] = (packed[-1][0], end)
        else:
            packed.append((start, end))
    return packed


def _trim(text: str, span: Span) -> Span:
    """Strip leading and trailing whitespace from a span.

    Args:
        text: The full text.
        span: ``(start, end)`` span.

    Returns:
        The trimmed span (unchanged when it is whitespace only).
    """
    start, end = span
    piece = text[start:end]
    stripped = piece.lstrip()
    if not stripped:
        return span
    lead = len(piece) - len(stripped)
    return (start + lead, start + lead + len(stripped.rstrip()))


def _to_chunks(
    text: str,
    file_path: str,
    kind: str,
    spans: Sequence[Span],
    trim: bool = False,
) -> List[Chunk]:
    """Build chunks from spans, dropping whitespace-only ones.

    Args:
        text: The full text.
        file_path: Path stored in every chunk.
        kind: Chunk kind label.
        spans: ``(start, end)`` spans in order.
        trim: Strip leading/trailing whitespace from each chunk.

    Returns:
        The chunks.
    """
    if trim:
        spans = [_trim(text, sp) for sp in spans]
    return [
        Chunk(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
            kind=kind,
            text=text[start:end],
        )
        for start, end in spans
        if text[start:end].strip()
    ]


def chunk_text(
    text: str, file_path: str, max_chunk_size: int, trim: bool = False
) -> List[Chunk]:
    """Chunk plain text by paragraph, line or word boundaries.

    Args:
        text: File content.
        file_path: Path stored in every chunk.
        max_chunk_size: Maximum chunk size in characters.
        trim: Strip leading/trailing whitespace from each chunk.

    Returns:
        The chunks, in file order.

    Raises:
        ValueError: If ``max_chunk_size`` is smaller than 1.
    """
    _check_size(max_chunk_size)
    spans = _split_span(text, 0, len(text), max_chunk_size)
    return _to_chunks(text, file_path, "text", spans, trim)


def _line_starts(text: str) -> List[int]:
    """Compute the offset at which each line starts.

    Args:
        text: The full text.

    Returns:
        ``starts[i]`` is the offset of line ``i + 1`` (1-based, as in
        ``ast``); a final entry equal to ``len(text)`` exists when the text
        ends with a newline.
    """
    return [0] + [m.end() for m in _NEWLINE_RE.finditer(text)]


def _line_end(starts: List[int], text: str, lineno: int) -> int:
    """Offset just after the given (1-based) line, including its newline.

    Args:
        starts: Line start table from ``_line_starts``.
        text: The full text.
        lineno: 1-based line number.

    Returns:
        The end offset of the line.
    """
    return starts[lineno] if lineno < len(starts) else len(text)


def _is_definition(node: ast.stmt) -> bool:
    """Tell whether a statement is a function or class definition.

    Args:
        node: An AST statement.

    Returns:
        True for ``def``, ``async def`` and ``class`` statements.
    """
    return isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    )


def _definition_start(
    node: ast.stmt, text: str, starts: List[int]
) -> int:
    """Offset of the first line of a definition.

    Includes decorators and directly preceding comment lines.

    Args:
        node: A function or class definition.
        text: The full text.
        starts: Line start table.

    Returns:
        The start offset of that line.
    """
    linenos = [node.lineno]
    linenos += [d.lineno for d in getattr(node, "decorator_list", [])]
    lineno = min(linenos)
    while lineno > 1:
        prev = text[starts[lineno - 2]:starts[lineno - 1]]
        if not prev.lstrip().startswith("#"):
            break
        lineno -= 1
    return starts[lineno - 1]


def _segments(
    tree: ast.Module, text: str, starts: List[int]
) -> List[Tuple[Span, ast.stmt]]:
    """Group top-level statements into contiguous segments.

    Each definition is its own segment; consecutive other statements form
    one segment. Gaps (blank lines, comments) belong to the segment that
    follows them, and trailing text belongs to the last segment.

    Args:
        tree: Parsed module.
        text: The full text.
        starts: Line start table.

    Returns:
        ``((start, end), first_node)`` per segment, covering the text.
    """
    groups: List[List[ast.stmt]] = []
    for node in tree.body:
        if _is_definition(node) or not groups or _is_definition(
            groups[-1][0]
        ):
            groups.append([node])
        else:
            groups[-1].append(node)
    segments: List[Tuple[Span, ast.stmt]] = []
    if not groups:
        return [((0, len(text)), ast.Pass())]
    prev_end = 0
    for i, group in enumerate(groups):
        last = group[-1]
        end_lineno = getattr(last, "end_lineno", None) or last.lineno
        end = _line_end(starts, text, end_lineno)
        if i == len(groups) - 1:
            end = len(text)
        end = max(end, prev_end)
        segments.append(((prev_end, end), group[0]))
        prev_end = end
    return segments


def _class_pieces(
    node: ast.ClassDef, span: Span, text: str, starts: List[int]
) -> List[Span]:
    """Split a class segment into its header and one piece per method.

    Args:
        node: The class definition.
        span: The class segment ``(start, end)``.
        text: The full text.
        starts: Line start table.

    Returns:
        Contiguous pieces covering the segment.
    """
    start, end = span
    children = [n for n in node.body if _is_definition(n)]
    if not children:
        return [span]
    cuts = [_definition_start(children[0], text, starts)]
    for child in children[:-1]:
        lineno = getattr(child, "end_lineno", None) or child.lineno
        cuts.append(_line_end(starts, text, lineno))
    pieces: List[Span] = []
    prev = start
    for cut in cuts:
        cut = min(max(cut, prev), end)
        if cut > prev:
            pieces.append((prev, cut))
        prev = cut
    if end > prev:
        pieces.append((prev, end))
    return pieces


def chunk_python(
    text: str, file_path: str, max_chunk_size: int, trim: bool = False
) -> List[Chunk]:
    """Chunk Python source along top-level definitions and methods.

    Falls back to ``chunk_text`` when the source cannot be parsed.

    Args:
        text: File content.
        file_path: Path stored in every chunk.
        max_chunk_size: Maximum chunk size in characters.
        trim: Strip leading/trailing whitespace from each chunk.

    Returns:
        The chunks, in file order.

    Raises:
        ValueError: If ``max_chunk_size`` is smaller than 1.
    """
    _check_size(max_chunk_size)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return chunk_text(text, file_path, max_chunk_size, trim)
    starts = _line_starts(text)
    spans: List[Span] = []
    for span, node in _segments(tree, text, starts):
        if span[1] - span[0] <= max_chunk_size:
            spans.append(span)
        elif isinstance(node, ast.ClassDef):
            pieces = _pack(
                _class_pieces(node, span, text, starts), max_chunk_size
            )
            for start, end in pieces:
                spans.extend(_split_span(text, start, end, max_chunk_size))
        else:
            spans.extend(_split_span(text, span[0], span[1], max_chunk_size))
    spans = _pack(spans, int(max_chunk_size * _PY_MERGE_FRACTION))
    return _to_chunks(text, file_path, "python", spans, trim)


def _advance_fence(line: str, fence: str) -> Tuple[str, bool]:
    """Track fenced code blocks one line at a time.

    Args:
        line: The current line.
        fence: The opening fence marker, or "" outside a code block.

    Returns:
        The new fence state and whether the line is part of a code
        block (including its opening and closing fence lines).
    """
    match = _FENCE_RE.match(line)
    if fence:
        if (
            match
            and match.group(1)[0] == fence[0]
            and len(match.group(1)) >= len(fence)
            and not match.group(2).strip()
        ):
            return "", True
        return fence, True
    if match:
        return match.group(1), True
    return "", False


def _markdown_sections(text: str) -> List[Span]:
    """Split markdown into sections at ATX headings outside code fences.

    Args:
        text: The full text.

    Returns:
        Contiguous ``(start, end)`` sections covering the text.
    """
    cuts: List[int] = []
    fence = ""
    offset = 0
    for line in text.split("\n"):
        fence, in_code = _advance_fence(line, fence)
        if not in_code and _HEADING_RE.match(line):
            cuts.append(offset)
        offset += len(line) + 1
    bounds = [0] + [c for c in cuts if c > 0] + [len(text)]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def chunk_markdown(
    text: str, file_path: str, max_chunk_size: int, trim: bool = False
) -> List[Chunk]:
    """Chunk markdown by heading sections.

    Oversized sections are split on paragraph boundaries, and sections
    shorter than 400 characters are merged into the following one when the
    result still fits.

    Args:
        text: File content.
        file_path: Path stored in every chunk.
        max_chunk_size: Maximum chunk size in characters.
        trim: Strip leading/trailing whitespace from each chunk.

    Returns:
        The chunks, in file order.

    Raises:
        ValueError: If ``max_chunk_size`` is smaller than 1.
    """
    _check_size(max_chunk_size)
    pieces: List[Span] = []
    for start, end in _markdown_sections(text):
        pieces.extend(_split_span(text, start, end, max_chunk_size))
    merged: List[Span] = []
    for start, end in pieces:
        if (
            merged
            and merged[-1][1] - merged[-1][0] < _MIN_SECTION_SIZE
            and end - merged[-1][0] <= max_chunk_size
        ):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return _to_chunks(text, file_path, "markdown", merged, trim)


def chunk_file(
    text: str, file_path: str, max_chunk_size: int, trim: bool = False
) -> List[Chunk]:
    """Chunk a file with the strategy matching its extension.

    Args:
        text: File content.
        file_path: Path stored in every chunk; its extension selects the
            chunker (``.py``, ``.md``/``.mdx``/``.rst``, otherwise text).
        max_chunk_size: Maximum chunk size in characters.
        trim: Strip leading/trailing whitespace from each chunk.

    Returns:
        The chunks, in file order.

    Raises:
        ValueError: If ``max_chunk_size`` is smaller than 1.
    """
    lower = file_path.lower()
    if lower.endswith(".py"):
        return chunk_python(text, file_path, max_chunk_size, trim)
    if lower.endswith(_MARKDOWN_SUFFIXES):
        return chunk_markdown(text, file_path, max_chunk_size, trim)
    return chunk_text(text, file_path, max_chunk_size, trim)


def markdown_heading_contexts(text: str, chunks: Sequence[Chunk]) -> List[str]:
    """Find the parent headings that precede each chunk of a markdown file.

    Args:
        text: The full file content.
        chunks: Chunks of that file, in file order.

    Returns:
        For each chunk, the titles of the headings that are open at its
        start (excluding headings located inside the chunk itself).
    """
    heads: List[Tuple[int, int, str]] = []
    fence = ""
    offset = 0
    for line in text.split("\n"):
        fence, in_code = _advance_fence(line, fence)
        if not in_code and _HEADING_RE.match(line):
            stripped = line.strip()
            level = len(stripped) - len(stripped.lstrip("#"))
            heads.append((offset, level, stripped.lstrip("#").strip()))
        offset += len(line) + 1
    contexts: List[str] = []
    for chunk in chunks:
        stack: List[Tuple[int, str]] = []
        for pos, level, title in heads:
            if pos >= chunk.first_character_index:
                break
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        contexts.append(" ".join(t for _, t in stack))
    return contexts
