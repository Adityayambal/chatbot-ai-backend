"""Recursive MathML -> LaTeX conversion.

Operates on BeautifulSoup Tag nodes (as produced by html.parser, which keeps MathML
tags intact). Naive tag stripping corrupts formulas (e.g. "s = ut + 1/2 at^2" becomes
"s=ut+12at2"), so every element is mapped to its LaTeX equivalent explicitly.
"""
from __future__ import annotations

from bs4 import NavigableString, Tag

_GREEK: dict[str, str] = {
    "α": "\\alpha", "β": "\\beta", "γ": "\\gamma", "Γ": "\\Gamma",
    "δ": "\\delta", "Δ": "\\Delta", "ε": "\\epsilon", "ζ": "\\zeta",
    "η": "\\eta", "θ": "\\theta", "Θ": "\\Theta", "ι": "\\iota",
    "κ": "\\kappa", "λ": "\\lambda", "Λ": "\\Lambda", "μ": "\\mu",
    "ν": "\\nu", "ξ": "\\xi", "Ξ": "\\Xi", "ο": "o",
    "π": "\\pi", "Π": "\\Pi", "ρ": "\\rho", "σ": "\\sigma",
    "Σ": "\\Sigma", "τ": "\\tau", "υ": "\\upsilon", "φ": "\\phi",
    "Φ": "\\Phi", "χ": "\\chi", "ψ": "\\psi", "Ψ": "\\Psi",
    "ω": "\\omega", "Ω": "\\Omega",
}

_FUNCTIONS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "lim", "exp"}

_MO_MAP: dict[str, str] = {
    "×": "\\times", "÷": "\\div", "−": "-", "±": "\\pm", "·": "\\cdot",
    "≤": "\\le", "≥": "\\ge", "≠": "\\ne", "→": "\\rightarrow",
    "⇌": "\\rightleftharpoons", "∞": "\\infty", "°": "^{\\circ}",
}

_ANNOTATION_TAGS = {"annotation", "annotation-xml"}


def _children(tag: Tag) -> list[Tag | NavigableString]:
    return [c for c in tag.contents if not (isinstance(c, Tag) and c.name in _ANNOTATION_TAGS)]


def _convert_node(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    return _convert_tag(node)


def _concat_children(tag: Tag) -> str:
    return "".join(_convert_node(c) for c in _children(tag))


def _text_content(tag: Tag) -> str:
    return "".join(str(c) for c in tag.contents if isinstance(c, NavigableString)).strip()


def _convert_mi(tag: Tag) -> str:
    text = _text_content(tag)
    if text in _GREEK:
        return _GREEK[text]
    if text.lower() in _FUNCTIONS:
        return f"\\{text.lower()}"
    return text


def _convert_mo(tag: Tag) -> str:
    text = _text_content(tag)
    return _MO_MAP.get(text, text)


def _convert_mfrac(tag: Tag) -> str:
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    if len(parts) != 2:
        return _concat_children(tag)
    num, den = parts
    return f"\\frac{{{_convert_tag(num)}}}{{{_convert_tag(den)}}}"


def _convert_msup(tag: Tag) -> str:
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    if len(parts) != 2:
        return _concat_children(tag)
    base, sup = parts
    return f"{{{_convert_tag(base)}}}^{{{_convert_tag(sup)}}}"


def _convert_msub(tag: Tag) -> str:
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    if len(parts) != 2:
        return _concat_children(tag)
    base, sub = parts
    return f"{{{_convert_tag(base)}}}_{{{_convert_tag(sub)}}}"


def _convert_msubsup(tag: Tag) -> str:
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    if len(parts) != 3:
        return _concat_children(tag)
    base, sub, sup = parts
    return f"{{{_convert_tag(base)}}}_{{{_convert_tag(sub)}}}^{{{_convert_tag(sup)}}}"


def _convert_msqrt(tag: Tag) -> str:
    return f"\\sqrt{{{_concat_children(tag)}}}"


def _convert_mroot(tag: Tag) -> str:
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    if len(parts) != 2:
        return _concat_children(tag)
    base, index = parts
    return f"\\sqrt[{_convert_tag(index)}]{{{_convert_tag(base)}}}"


_ACCENT_MAP = {
    "→": ("\\vec", True),
    "¯": ("\\overline", True),
    "^": ("\\hat", True),
}


def _convert_mover(tag: Tag) -> str:
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    if len(parts) != 2:
        return _concat_children(tag)
    base, accent = parts
    accent_text = accent.get_text().strip()
    mapped = _ACCENT_MAP.get(accent_text)
    base_latex = _convert_tag(base)
    if mapped:
        return f"{mapped[0]}{{{base_latex}}}"
    return f"\\overset{{{_convert_tag(accent)}}}{{{base_latex}}}"


def _convert_munder(tag: Tag) -> str:
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    if len(parts) != 2:
        return _concat_children(tag)
    base, under = parts
    return f"\\underset{{{_convert_tag(under)}}}{{{_convert_tag(base)}}}"


def _convert_munderover(tag: Tag) -> str:
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    if len(parts) != 3:
        return _concat_children(tag)
    base, under, over = parts
    return f"{{{_convert_tag(base)}}}_{{{_convert_tag(under)}}}^{{{_convert_tag(over)}}}"


def _convert_mfenced(tag: Tag) -> str:
    open_ch = tag.get("open", "(")
    close_ch = tag.get("close", ")")
    seps = tag.get("separators", ",")
    parts = [c for c in _children(tag) if isinstance(c, Tag)]
    pieces = [_convert_tag(p) for p in parts]
    if not seps:
        seps = ","
    out = ""
    for i, piece in enumerate(pieces):
        if i == 0:
            out += piece
        else:
            sep = seps[min(i - 1, len(seps) - 1)]
            out += sep + piece
    return f"\\left{open_ch}{out}\\right{close_ch}"


def _convert_mtable(tag: Tag) -> str:
    rows = []
    for tr in tag.find_all("mtr", recursive=False):
        cells = [_convert_tag(td) for td in tr.find_all("mtd", recursive=False)]
        rows.append(" & ".join(cells))
    body = " \\\\ ".join(rows)
    return f"\\begin{{matrix}} {body} \\end{{matrix}}"


def _convert_tag(tag: Tag) -> str:
    name = tag.name
    if name in ("math", "mrow", "mstyle", "semantics"):
        return _concat_children(tag)
    if name == "mi":
        return _convert_mi(tag)
    if name == "mn":
        return _text_content(tag)
    if name == "mo":
        return _convert_mo(tag)
    if name == "mtext":
        return f"\\text{{{_text_content(tag)}}}"
    if name == "mspace":
        return "\\ "
    if name == "mfrac":
        return _convert_mfrac(tag)
    if name == "msup":
        return _convert_msup(tag)
    if name == "msub":
        return _convert_msub(tag)
    if name == "msubsup":
        return _convert_msubsup(tag)
    if name == "msqrt":
        return _convert_msqrt(tag)
    if name == "mroot":
        return _convert_mroot(tag)
    if name == "mover":
        return _convert_mover(tag)
    if name == "munder":
        return _convert_munder(tag)
    if name == "munderover":
        return _convert_munderover(tag)
    if name == "mfenced":
        return _convert_mfenced(tag)
    if name == "mtable":
        return _convert_mtable(tag)
    if name in ("mtr", "mtd"):
        return _concat_children(tag)
    # unknown: concatenate children
    return _concat_children(tag)


def mathml_to_latex(tag: Tag) -> str:
    """Convert a <math>...</math> BeautifulSoup Tag into a LaTeX string (no $ wrapper)."""
    return _convert_tag(tag).strip()
