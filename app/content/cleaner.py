"""HTML -> clean Markdown-ish text, for lesson content coming from the content-bundle API."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

from app.content.mathml import mathml_to_latex

_BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

_SUB_MAP = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇",
            "8": "₈", "9": "₉", "+": "₊", "-": "₋"}
_SUP_MAP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
            "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻"}

_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

_GREETING_RE = re.compile(
    r"^\s*hi\s+friends[.,]?\s*welcome to top scholars[,]?\s*the smart learning app[.,]?\s*",
    re.IGNORECASE,
)
_OUTRO_RE = re.compile(
    r"\s*(thanks?|thank you)\s+for watching[.,]?\s*(see you in the next video[.,]?)?\s*$",
    re.IGNORECASE,
)


@dataclass
class ImageRef:
    n: int
    url: str
    alt: str = ""


@dataclass
class CleanResult:
    text: str
    images: list[ImageRef] = field(default_factory=list)


def _finalize(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    # Collapse/trim whitespace per line, but preserve leading indentation (list nesting).
    cleaned_lines = []
    for line in text.split("\n"):
        indent_len = len(line) - len(line.lstrip(" "))
        rest = _MULTI_SPACE_RE.sub(" ", line[indent_len:]).rstrip()
        cleaned_lines.append(line[:indent_len] + rest if rest else "")
    text = "\n".join(cleaned_lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _inline(node: Tag | NavigableString, images: list[ImageRef], counter: list[int]) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    name = node.name
    if name == "math":
        latex = mathml_to_latex(node)
        display = node.get("display") == "block"
        return f"$${latex}$$" if display else f"${latex}$"
    if name in ("strong", "b"):
        inner = "".join(_inline(c, images, counter) for c in node.children).strip()
        return f"**{inner}**" if inner else ""
    if name in ("em", "i"):
        inner = "".join(_inline(c, images, counter) for c in node.children).strip()
        return f"_{inner}_" if inner else ""
    if name == "br":
        return "\n"
    if name == "sub":
        return _render_subsup(node, images, counter, sup=False)
    if name == "sup":
        return _render_subsup(node, images, counter, sup=True)
    if name == "img":
        counter[0] += 1
        n = counter[0]
        src = node.get("src", "")
        alt = node.get("alt", "")
        images.append(ImageRef(n=n, url=src, alt=alt))
        return f"[Figure {n}]"
    # unknown inline-ish / passthrough tags (span, a, ...): just descend
    return "".join(_inline(c, images, counter) for c in node.children)


def _render_subsup(node: Tag, images: list[ImageRef], counter: list[int], *, sup: bool) -> str:
    text = "".join(_inline(c, images, counter) for c in node.children).strip()
    mapping = _SUP_MAP if sup else _SUB_MAP
    if text and all(ch in mapping for ch in text):
        return "".join(mapping[ch] for ch in text)
    return f"^({text})" if sup else f"_({text})"


def _render_list(node: Tag, images: list[ImageRef], counter: list[int], indent: int) -> str:
    lines: list[str] = []
    ordered = node.name == "ol"
    prefix = "  " * indent
    idx = 0
    for li in node.find_all("li", recursive=False):
        idx += 1
        marker = f"{idx}." if ordered else "-"
        nested_lists = [c for c in li.find_all(["ul", "ol"], recursive=False)]
        own_parts = [
            _inline(c, images, counter)
            for c in li.children
            if not (isinstance(c, Tag) and c.name in ("ul", "ol"))
        ]
        own_text = "".join(own_parts).strip()
        lines.append(f"{prefix}{marker} {own_text}")
        for nl in nested_lists:
            lines.append(_render_list(nl, images, counter, indent + 1))
    return "\n".join(lines)


def _render_table(node: Tag, images: list[ImageRef], counter: list[int]) -> str:
    rows: list[list[str]] = []
    for tr in node.find_all("tr"):
        cells: list[str] = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            text = "".join(_inline(c, images, counter) for c in cell.children).strip()
            text = text.replace("|", "\\|").replace("\n", " ")
            try:
                colspan = max(int(cell.get("colspan", 1)), 1)
            except (TypeError, ValueError):
                colspan = 1
            cells.extend([text] * colspan)
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (ncols - len(r)))
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _process_block_children(
    container: Tag | BeautifulSoup, images: list[ImageRef], counter: list[int]
) -> list[str]:
    blocks: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        combined = "".join(pending).strip()
        if combined:
            blocks.append(combined)
        pending.clear()

    for child in container.children:
        if isinstance(child, NavigableString):
            if str(child).strip():
                pending.append(str(child))
            continue
        name = child.name
        if name in _HEADING_TAGS:
            flush()
            heading_text = _inline(child, images, counter).strip()
            if heading_text:
                blocks.append(f"### {heading_text}")
        elif name in ("ul", "ol"):
            flush()
            rendered = _render_list(child, images, counter, 0)
            if rendered:
                blocks.append(rendered)
        elif name == "table":
            flush()
            rendered = _render_table(child, images, counter)
            if rendered:
                blocks.append(rendered)
        elif name in ("p", "div"):
            if name == "div" and child.find(list(_BLOCK_TAGS)):
                flush()
                blocks.extend(_process_block_children(child, images, counter))
            else:
                flush()
                text = _inline(child, images, counter).strip()
                if text:
                    blocks.append(text)
        elif name == "br":
            pending.append("\n")
        else:
            pending.append(_inline(child, images, counter))
    flush()
    return blocks


def clean_html(html: str) -> CleanResult:
    soup = BeautifulSoup(html or "", "html.parser")
    images: list[ImageRef] = []
    counter = [0]
    blocks = _process_block_children(soup, images, counter)
    text = _finalize("\n\n".join(blocks))
    return CleanResult(text=text, images=images)


def clean_transcript(text: str) -> str:
    text = _GREETING_RE.sub("", text or "")
    text = _OUTRO_RE.sub("", text)
    return _finalize(text)


def normalize_name(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()
