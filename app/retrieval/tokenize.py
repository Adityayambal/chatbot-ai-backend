"""Tokeniser shared by indexing and query-time retrieval. Deliberately simple (BM25, not
an embeddings model) but careful not to mangle unit symbols a physics/chem textbook uses."""
from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "if", "then", "than", "so", "that", "this", "these", "those",
    "it", "its", "as", "by", "with", "from", "about", "into", "over", "after", "before", "between",
    "during", "above", "below", "up", "down", "out", "off", "again", "further", "once", "there",
    "here", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "too", "very", "can",
    "will", "just", "do", "does", "did", "doing", "have", "has", "had", "having", "i", "you", "he",
    "she", "we", "they", "them", "his", "her", "their", "our", "your", "what", "which", "who", "whom",
}

# A degree symbol followed by a unit letter (°C, °F), or an alnum run with an optional
# "/unit" suffix (m/s, km/hr) — in that priority order so "°c" isn't split into "°" + "c".
_TOKEN_RE = re.compile(r"°[a-z]|[^\W_]+(?:/[^\W_]+)?", re.UNICODE)

_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(token: str) -> str:
    if token.isdigit():
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        if match in _STOPWORDS:
            continue
        tokens.append(_stem(match))
    return tokens
