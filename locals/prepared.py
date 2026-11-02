"""A prepared prompt: a rendered chat prompt as the model reads it, and what follows from its images.

The server renders the conversation with the model's template; this module turns that text into the token ids the
model reads (each image's one placeholder expanded to the image's own tokens, GLOSSARY.md "Image prompt")."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from . import vision

# text -> (the text of the longest remembered stream it continues, the ids the model saw for it), or None
Remembered = Callable[[str], Optional[Tuple[str, Sequence[int]]]]


@dataclass(frozen=True)
class PreparedPrompt:
    """``text`` as the model reads it: ``tokens``, with each of ``pictures``' tokens at its ``spans``."""

    text: Optional[str]  # (None: a /v1/completions prompt given as token ids)
    tokens: List[int]
    pictures: tuple = ()
    spans: tuple = ()
    placed: Tuple[int, ...] = ()  # an image prompt's tokens with one placeholder per image (no images: empty)
    tokenizer: object = field(default=None, repr=False, compare=False)

    @classmethod
    def of(cls, text: str, tokenizer, pictures: Sequence[vision.Picture] = (),
           remembered: Optional[Remembered] = None) -> "PreparedPrompt":
        """``text`` (one ``<|image_pad|>`` per picture) as token ids. Without pictures it goes on from the remembered
        stream it continues (the ids the model saw, which the text may not encode to); an image prompt reads none."""
        hf = tokenizer._tokenizer
        if pictures:
            fam = pictures[0].family
            placed = list(hf.encode(text, add_special_tokens=False))
            token, boi, eoi = vision.prompt_tokens(tokenizer, fam)
            tokens, spans = vision.expand(placed, token, pictures, boi, eoi)
            return cls(text, tokens, tuple(pictures), tuple(spans), tuple(placed), tokenizer)
        seen = remembered(text) if remembered is not None else None
        if seen is None:
            tokens = list(hf.encode(text, add_special_tokens=False))
        else:
            tokens = list(seen[1]) + list(hf.encode(text[len(seen[0]):], add_special_tokens=False))
        return cls(text, tokens, tokenizer=tokenizer)

    @property
    def rememberable(self) -> bool:
        """Whether the reply's ids may be remembered for the next request's text (PreparedPrompt.of reads them back
        in place of the text's): not an image prompt's, whose ids hold its images' tokens, nor one with no text."""
        return not self.pictures and self.text is not None

    def checkpoint_at(self, prefix: str) -> Optional[int]:
        """Where ``prefix``, a text prefix of the prompt, ends in its tokens (an image's tokens before it count: its
        prefix costs the tower and their prefill), or None: no prefix, the whole prompt, or under 1024 tokens (a
        short prefix is cheaper to read again than to keep)."""
        if self.text is None or (len(prefix) < 2048 and not self.pictures) or not self.text.startswith(prefix):
            return None
        ids = self.tokenizer._tokenizer.encode(prefix, add_special_tokens=False)
        placed = self.placed or self.tokens
        # where the prompt's tokens and these part (the last token may merge with what follows): one pass
        # (backing off a token at a time cost seconds on a 60k-token prompt)
        n = next((k for k, (a, b) in enumerate(zip(placed, ids)) if a != b), min(len(placed), len(ids)))
        at = vision.expanded_at(n, self.spans, self.pictures)
        return at if at >= 1024 and n < len(placed) else None
