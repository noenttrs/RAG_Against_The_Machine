"""Tokenizer shared by indexing and querying."""

import re
from functools import lru_cache
from typing import List, Tuple

_RAW_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z](?![A-Z][a-z]))|(?<=[A-Z])(?=[A-Z][a-z])"
)
STOPWORDS = frozenset(
    "the a an is are of to in and or for on with what how does do which "
    "when where why be by as it this that from at can i you".split()
)


def _stem(token: str) -> str:
    """Strip a trailing plural ``s`` from long enough tokens.

    Args:
        token: A lowercase token.

    Returns:
        The stemmed token.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@lru_cache(maxsize=200_000)
def _expand(raw: str) -> Tuple[str, ...]:
    """Expand one raw token into its full form and its sub-parts.

    Args:
        raw: A raw token matched by ``[A-Za-z0-9_]+``.

    Returns:
        The normalized terms (stopwords removed, stemmed).
    """
    terms: List[str] = []
    if len(raw) >= 2:
        terms.append(raw.lower())
    parts: List[str] = []
    for piece in raw.split("_"):
        if piece:
            parts.extend(_CAMEL_RE.split(piece))
    if len(parts) > 1:
        terms.extend(p.lower() for p in parts if len(p) >= 2)
    return tuple(_stem(t) for t in terms if t not in STOPWORDS)


def tokenize(text: str, bigrams: bool = False) -> List[str]:
    """Tokenize text into normalized terms.

    Identifiers are emitted whole and split on underscores and camelCase.

    Args:
        text: The text to tokenize.
        bigrams: Also emit bigrams of adjacent words.

    Returns:
        The list of terms (with repetitions).
    """
    out: List[str] = []
    prev = ""
    for raw in _RAW_RE.findall(text):
        terms = _expand(raw)
        out.extend(terms)
        if bigrams:
            head = terms[0] if terms and terms[0] == _stem(raw.lower()) else ""
            if head and prev:
                out.append(prev + " " + head)
            prev = head
    return out
