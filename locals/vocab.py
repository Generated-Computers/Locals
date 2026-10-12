"""Token byte strings and the vocabulary byte trie.

The trie is an out-arborescence rooted at the empty prefix: every node has one
root path (a mangrove, Thm A23.1). Nodes are stored level by level so that
products with an automaton can be propagated one depth at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def _bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_BYTE_DECODER = {c: b for b, c in _bytes_to_unicode().items()}


# Control tokens of tool-call and thinking formats (locals.formats): kept in the
# vocabulary with their text even when special, so grammars can place them.
FORMAT_TOKENS = frozenset({"[TOOL_CALLS]", "[ARGS]", "[CALL_ID]", "<|python_tag|>", "<tool_call>", "</tool_call>",
                           "<think>", "</think>", "<|tool_call>", "<tool_call|>", '<|"|>', "<|channel>", "<channel|>",
                           "<|think|>", "[THINK]", "[/THINK]",
                           # MiniCPM5 writes its calls with special tokens: kept out, its calls could only be spelled
                           # letter by letter, which it never does (it answered in prose instead of calling)
                           "<function", "</function>", "<param", "</param>"})


def _is_byte_level(hf_tokenizer, pieces) -> bool:
    """Whether token pieces are byte-level BPE (GPT-2's byte-to-unicode map: Llama 3,
    Qwen, Ministral 3) rather than SentencePiece pieces ("▁" for a space,
    <0xNN> byte fallback: Gemma, Mistral 7B). The tokenizer's decoder says so;
    without one, the pieces' own markers decide."""
    import json

    try:
        dec = json.loads(hf_tokenizer.backend_tokenizer.to_str()).get("decoder") or {}
    except Exception:
        dec = {}

    def types(d):
        if isinstance(d, dict):
            yield d.get("type")
            for v in d.values():
                yield from types(v)
        elif isinstance(d, list):
            for v in d:
                yield from types(v)

    kinds = set(types(dec))
    if "ByteLevel" in kinds:
        return True
    if kinds & {"Metaspace", "ByteFallback", "Replace"}:
        return False
    sample = [p for p in pieces if p][:5000]
    spm = sum(p.startswith("▁") for p in sample)
    gpt2 = sum(p.startswith("Ġ") for p in sample)
    return gpt2 > spm


@dataclass
class TokenVocab:
    token_bytes: List[Optional[bytes]]  # None for special or padding ids
    eos_ids: List[int]
    n_model: int  # logits width

    @classmethod
    def from_hf(cls, hf_tokenizer, n_model: int, eos_ids: Sequence[int]) -> "TokenVocab":
        n = len(hf_tokenizer)
        special = set(hf_tokenizer.all_special_ids)
        # tokens the tokenizer marks special without listing them there (Llama 3's 254 control tokens:
        # <|start_header_id|>, <|end_of_text|>, the reserved ones): control tokens, never text a grammar
        # may write (a JSON value could otherwise hold a role header)
        special |= {i for i, t in (getattr(hf_tokenizer, "added_tokens_decoder", None) or {}).items() if getattr(t, "special", False)}
        # Added tokens that are not special (Qwen's <tool_call>, </think>, ...)
        # are ordinary text to the tokenizer: it matches them before byte
        # merges, so they get their surface bytes and constraints can allow
        # them where a literal expects that text. Tool-call control tokens are
        # included even where the tokenizer marks them special (Ministral's
        # [TOOL_CALLS]): a model emits them as one token, never letter by letter.
        added_all = hf_tokenizer.get_added_vocab()
        added = {i: t for t, i in added_all.items() if i not in special or t in FORMAT_TOKENS}
        special = special - set(added)
        pieces = hf_tokenizer.convert_ids_to_tokens(list(range(n)))
        byte_level = _is_byte_level(hf_tokenizer, pieces)
        out: List[Optional[bytes]] = [None] * n_model
        for i, p in enumerate(pieces):
            if i in special or p is None:
                continue
            if i in added:
                out[i] = added[i].encode("utf-8")
                continue
            if byte_level:
                try:
                    out[i] = bytes(_BYTE_DECODER[c] for c in p)
                except KeyError:
                    out[i] = None
            else:  # SentencePiece conventions
                if p.startswith("<0x") and p.endswith(">") and len(p) == 6:
                    out[i] = bytes([int(p[3:5], 16)])
                else:
                    out[i] = p.replace("▁", " ").encode("utf-8")
        return cls(out, list(eos_ids), n_model)


@dataclass
class ByteTrie:
    """Level-ordered trie. Level d holds the nodes at depth d.

    ``parent[d][j]`` is the index at level d-1 of node j's parent and
    ``byte[d][j]`` its last byte. ``tok_level/tok_index`` locate each token's
    node; tokens with no bytes have level -1.
    """

    parent: List[np.ndarray]
    byte: List[np.ndarray]
    tok_level: np.ndarray
    tok_index: np.ndarray

    @property
    def n_nodes(self) -> int:
        return sum(len(p) for p in self.parent)

    @property
    def depth(self) -> int:
        return len(self.parent) - 1

    @classmethod
    def build(cls, vocab: TokenVocab) -> "ByteTrie":
        ids: List[Dict[bytes, int]] = [{b"": 0}]
        parents: List[List[int]] = [[-1]]
        bytes_: List[List[int]] = [[0]]
        n = len(vocab.token_bytes)
        tok_level = np.full(n, -1, dtype=np.int32)
        tok_index = np.full(n, -1, dtype=np.int32)
        for t, tb in enumerate(vocab.token_bytes):
            if not tb:
                continue
            while len(ids) <= len(tb):
                ids.append({})
                parents.append([])
                bytes_.append([])
            prev = 0
            for d in range(1, len(tb) + 1):
                key = tb[:d]
                j = ids[d].get(key)
                if j is None:
                    j = len(parents[d])
                    ids[d][key] = j
                    parents[d].append(prev)
                    bytes_[d].append(tb[d - 1])
                prev = j
            tok_level[t] = len(tb)
            tok_index[t] = prev
        return cls(
            [np.asarray(p, dtype=np.int32) for p in parents],
            [np.asarray(b, dtype=np.int32) for b in bytes_],
            tok_level,
            tok_index,
        )
