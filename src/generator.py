"""Answer generation with a local causal language model."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import Chunk, MinimalSource

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the vLLM "
    "codebase. Answer ONLY using the provided context. If the context "
    "does not contain the answer, say you could not find it. Be concise "
    "(2-5 sentences) and cite file names when useful."
)
NO_CONTEXT = "No context retrieved."
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def format_block(index: int, chunk: Chunk, text: Optional[str] = None) -> str:
    """Format one numbered context block.

    Args:
        index: One-based rank of the chunk.
        chunk: The chunk.
        text: Text to use instead of the chunk text (truncation).

    Returns:
        The block text.
    """
    body = chunk.text if text is None else text
    return (
        f"[{index}] {chunk.file_path} "
        f"(chars {chunk.first_character_index}-"
        f"{chunk.last_character_index})\n{body}"
    )


def build_messages(
    question: str, contexts: List[Chunk]
) -> List[Dict[str, str]]:
    """Build the chat messages for a question and its contexts.

    Args:
        question: The question.
        contexts: Retrieved chunks in rank order.

    Returns:
        System and user messages.
    """
    if contexts:
        blocks = "\n\n".join(
            format_block(i, c) for i, c in enumerate(contexts, 1)
        )
    else:
        blocks = NO_CONTEXT
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{blocks}\n\nQuestion: {question}"},
    ]


def strip_thinking(text: str) -> str:
    """Remove ``<think>`` sections and surrounding whitespace.

    Args:
        text: Raw model output.

    Returns:
        The cleaned answer.
    """
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def load_source_text(
    source: MinimalSource,
    root: str,
    cache: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Read the text span designated by a source from disk.

    Args:
        source: The source span.
        root: Directory against which ``file_path`` is resolved.
        cache: Optional dict caching file contents between calls.

    Returns:
        The text, or None if the file cannot be read.
    """
    key = source.file_path
    if cache is not None and key in cache:
        content = cache[key]
    else:
        try:
            with open(Path(root, key), "rb") as handle:
                content = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        if cache is not None:
            cache[key] = content
    return content[
        source.first_character_index:source.last_character_index
    ]


class AnswerGenerator:
    """Generate answers from retrieved contexts with a causal LM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_context_tokens: int = 3000,
        max_new_tokens: int = 256,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        """Load the model and tokenizer.

        Args:
            model_name: Hugging Face model name.
            max_context_tokens: Token budget for the context blocks.
            max_new_tokens: Maximum number of generated tokens.
            model: Optional pre-built model (skips loading).
            tokenizer: Optional pre-built tokenizer (skips loading).
        """
        import torch

        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        if model is None or tokenizer is None:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            dtype = torch.float32
            if device == "cuda":
                dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype
            )
            model.to(device)
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    def _count(self, text: str) -> int:
        """Count the tokens of a text.

        Args:
            text: The text.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def fit_contexts(self, contexts: List[Chunk]) -> List[Chunk]:
        """Keep the leading chunks fitting the context token budget.

        The first chunk that does not fit is truncated if room remains.

        Args:
            contexts: Retrieved chunks in rank order.

        Returns:
            The (possibly truncated) chunks that fit.
        """
        kept: List[Chunk] = []
        used = 0
        for i, chunk in enumerate(contexts, 1):
            cost = self._count(format_block(i, chunk)) + 2
            if used + cost <= self.max_context_tokens:
                kept.append(chunk)
                used += cost
                continue
            room = (
                self.max_context_tokens - used
                - self._count(format_block(i, chunk, "")) - 2
            )
            if room > 0:
                ids = self.tokenizer.encode(
                    chunk.text, add_special_tokens=False
                )[:room]
                kept.append(
                    chunk.model_copy(
                        update={"text": self.tokenizer.decode(ids)}
                    )
                )
            break
        return kept

    def generate(self, question: str, contexts: List[Chunk]) -> str:
        """Answer a question from its contexts.

        Args:
            question: The question.
            contexts: Retrieved chunks in rank order.

        Returns:
            The generated answer.
        """
        import torch

        messages = build_messages(
            question, self.fit_contexts(contexts)
        )
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_ids = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return strip_thinking(text)
