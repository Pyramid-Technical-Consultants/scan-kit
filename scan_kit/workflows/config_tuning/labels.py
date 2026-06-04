"""Display-only label formatting for XML config tuning UI."""

from __future__ import annotations

import re

# Tokens that should not be title-cased blindly.
_SPECIAL_TOKENS: dict[str, str] = {
    "ic": "IC",
    "ic1": "IC1",
    "ic2": "IC2",
    "ic3": "IC3",
    "hcc": "HCC",
    "mev": "MeV",
    "mu": "MU",
    "gp": "gP",
    "hv": "HV",
    "mm": "mm",
    "iso": "ISO",
    "xml": "XML",
    "dcs": "DCS",
    "sc": "SC",
    "rci": "RCI",
    "k0": "K0",
    "k1": "K1",
    "k2": "K2",
    "k3": "K3",
    "k_mu": "K MU",
}

_SMALL_WORDS = frozenset({"a", "an", "and", "at", "by", "for", "in", "of", "or", "per", "to", "vs"})

_INDEX_SUFFIX_RE = re.compile(r"^(.+) \[(\d+)\]$")
_ROWS_SUFFIX_RE = re.compile(r"^(.+) \((\d+) rows\)$")
_K_COEFF_RE = re.compile(r"^k(\d+)$", re.IGNORECASE)


def _humanize_token(token: str, *, is_first: bool) -> str:
    if not token:
        return token

    lower = token.lower()
    if lower in _SPECIAL_TOKENS:
        return _SPECIAL_TOKENS[lower]

    k_match = _K_COEFF_RE.match(token)
    if k_match:
        return f"K{k_match.group(1)}"

    if token.isupper() and len(token) <= 4:
        return token

    if not is_first and lower in _SMALL_WORDS:
        return lower

    if token.isdigit():
        return token

    return token[:1].upper() + token[1:].lower()


def humanize_xml_label(name: str) -> str:
    """Convert an XML tag or attribute name into a readable UI label."""
    text = name.strip()
    if text.startswith("@"):
        text = text[1:]

    rows_suffix = ""
    rows_match = _ROWS_SUFFIX_RE.match(text)
    if rows_match:
        text = rows_match.group(1)
        rows_suffix = f" ({rows_match.group(2)} rows)"

    index_suffix = ""
    index_match = _INDEX_SUFFIX_RE.match(text)
    if index_match:
        text = index_match.group(1)
        index_suffix = f" [{index_match.group(2)}]"

    if text.lower() == "value":
        return "Value" + index_suffix + rows_suffix

    if " / " in text:
        return (
            " / ".join(humanize_xml_label(part.strip()) for part in text.split(" / "))
            + rows_suffix
            + index_suffix
        )

    parts = [part for part in text.split("_") if part]
    if not parts:
        return name

    humanized = " ".join(
        _humanize_token(part, is_first=(idx == 0))
        for idx, part in enumerate(parts)
    )
    return humanized + index_suffix + rows_suffix
