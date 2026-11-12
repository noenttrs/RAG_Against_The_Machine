"""Tests for the tokenizer."""

from src.tokenizer import tokenize


def test_camel_case() -> None:
    """CamelCase identifiers are kept whole and split."""
    toks = tokenize("FusedMoEActivationFormat")
    for t in ("fusedmoeactivationformat", "fused", "moe", "activation",
              "format"):
        assert t in toks


def test_snake_case() -> None:
    """Snake case identifiers are kept whole and split."""
    toks = tokenize("max_num_seqs")
    assert "max_num_seq" in toks or "max_num_seqs" in toks
    assert {"max", "num", "seq"} <= set(toks)


def test_stopwords_and_stemming() -> None:
    """Stopwords vanish and plurals are stemmed identically."""
    assert tokenize("what is the") == []
    assert tokenize("models") == tokenize("model")
    assert tokenize("class") == ["class"]


def test_empty() -> None:
    """Empty or symbol-only text gives no tokens."""
    assert tokenize("") == []
    assert tokenize("!!! ???") == []
