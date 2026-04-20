"""Pre-embedding text normalization: ligature fix + publisher-specific
boilerplate stripping + skip-to-abstract heuristic. Patterns are
conservative and line-anchored so inline mentions are preserved."""
from __future__ import annotations
import re

# PDF extraction leaves ligatures and combining-character copyright
# glyphs; fix so regex patterns can match plain text.
_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
})
_MULTI_CHAR_FIXES = [("C\u20dd", "©"), ("\u24b8", "©"), ("\u24d2", "c")]

# All patterns line-anchored unless noted. Grouped by source for
# readability; one flat list at runtime.
_BUILTIN: list[tuple[re.Pattern, str]] = [
    # --- EconStor / SSRN / CFR preprint catalog ---
    (re.compile(r"(?im)^[\t ]*Electronic\s+copy\s+available\s+at:\s*\S+[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*SSRN[\s-]*Electronic\s+Journal\s*$"), ""),
    (re.compile(r"(?im)^[\t ]*Provided\s+in\s+Cooperation\s+with:.*?(?=\n\s*\n|\Z)", re.DOTALL), ""),
    (re.compile(r"(?im)^[\t ]*Suggested\s+Citation:.*?(?=\n\s*\n|\Z)", re.DOTALL), ""),
    (re.compile(r"(?im)^[\t ]*This\s+Version\s+is\s+available\s+at:\s*\S+[ \t]*$"), ""),
    (re.compile(r"(?s)Standard-Nutzungsbedingungen:.*?(?=\n\s*\n|\Z)"), ""),
    (re.compile(r"(?is)(?<![A-Za-z])Terms\s+of\s+use:.*?(?=\n\s*\n|\Z)"), ""),
    (re.compile(r"(?im)^[\t ]*All\s+rights\s+reserved\.?[ \t]*$"), ""),
    # --- JSTOR catalog page ---
    (re.compile(r"(?im)^[\t ]*This content downloaded from\s+\d{1,3}(?:\.\d{1,3}){3}.*$"), ""),
    (re.compile(r"(?im)^[\t ]*Stable URL:\s*https?://www\.jstor\.org/stable/\S+[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*All use subject to\s+https?://about\.jstor\.org/terms[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*Your use of the JSTOR archive.*$"), ""),
    (re.compile(r"(?s)JSTOR is a not-for-profit service that helps scholars.*?support@jstor\.org\."), ""),
    (re.compile(r"(?im)^[\t ]*.+ are collaborating with JSTOR to digitize.*$"), ""),
    # --- Elsevier / ScienceDirect ---
    (re.compile(r"(?im)^[\t ]*Contents lists available at (?:SciVerse )?ScienceDirect[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*journal homepage:\s*www\.elsevier\.com/locate/\S+[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*Article history:[ \t]*$"), ""),
    (re.compile(r"(?m)^[\t ]*ARTICLE IN PRESS[ \t]*$"), ""),  # caps are the signal
    (re.compile(r"(?im)^[\t ]*(?:[a-z]\s+){4,}[a-z][ \t]*$"), ""),  # spaced "a b s t r a c t"
    (re.compile(r"(?im)^[\t ]*(?:\d{4}-\d{3,4}X?/[^A-Za-z\n]*)?(?:©|&|r)\s*\d{4}\s*Elsevier B\.V\..*$"), ""),
    # --- Oxford University Press ---
    (re.compile(r"(?im)^[\t ]*(?:All rights reserved\.\s*)?For [Pp]ermissions,? please e-mail:\s*journals\.permissions@oup\.com\.?[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*journals\.permissions@oup\.com\.?[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*©\s*The Authors?(?:\(s\))?\s+\d{4}\..*Published by Oxford University Press.*$", re.DOTALL), ""),
    # --- INFORMS ---
    (re.compile(r"(?im)^[\t ]*https?://pubsonline\.informs\.org[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*INFORMS is located in Maryland,\s*USA[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*Publisher:\s*Institute for Operations Research.*$"), ""),
    (re.compile(r"(?im)^[\t ]*ISSN\s+\d{4}-\d{4}\s*\(print\),\s*ISSN\s+\d{4}-\d{4}\s*\(online\)[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*Copyright:?\s*©?\s*\d{4}\s*,?\s*INFORMS\.?[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*Please scroll down for article.*$"), ""),
    (re.compile(r"(?im)^[\t ]*Full terms and conditions of use:\s*https?://pubsonline\.informs\.org.*$"), ""),
    (re.compile(r"(?im)^[\t ]*Publication details, including instructions for authors.*$"), ""),
    (re.compile(r"(?s)With\s+[\d,]+\s*members from.*?strategic visions and achieve better outcomes\."), ""),
    # --- Taylor & Francis / Crossmark ---
    (re.compile(r"(?im)^[\t ]*(?:View Crossmark data|View related articles|Submit your article to this journal)[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*Full Terms & Conditions of access and use can be found at[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*To link to this article:\s*https?://doi\.org/\S+[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*This article may be used only for the purposes of research.*$"), ""),
    # --- Wiley / American Finance Association ---
    (re.compile(r"(?im)^[\t ]*Published by:\s*Wiley for the American Finance Association[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*©\s*\d{4}\s+(?:the\s+)?American Finance Association[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*The Journal of Finance[®R]?[ \t]*$"), ""),
    # --- Econometric Society ---
    (re.compile(r"(?im)^[\t ]*Published by:\s*The Econometric Society[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*©\s*\d{4}\s+The Econometric Society[ \t]*$"), ""),
    # --- JFQA (Cambridge / Foster School) ---
    (re.compile(r"(?m)^[\t ]*JOURNAL OF FINANCIAL AND QUANTITATIVE ANALYSIS[ \t]*$"), ""),  # caps are the signal
    (re.compile(r"(?m)^[\t ]*MICHAEL G\. FOSTER SCHOOL OF BUSINESS[ \t]*$"), ""),  # caps are the signal
    # --- NBER address block ---
    (re.compile(r"(?m)^[\t ]*NATIONAL BUREAU OF ECONOMIC RESEARCH[ \t]*$"), ""),  # caps are the signal
    (re.compile(r"(?im)^[\t ]*\d{4}\s+Massachusetts Avenue[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*Cambridge,\s*MA\s*02138[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*(?:CFR|NBER|SSRN|HEC|CEPR|IZA)\s+Working\s+Paper(?:\s+Series)?(?:[\s,.:]+(?:No\.?\s*)?[\d\.\-\[\]\w]*)?[ \t]*$"), ""),
    # --- Generic ---
    (re.compile(r"(?im)^[\t ]*(?:DOI|doi):\s*10\.\d{4,9}/\S+[ \t]*$"), ""),
    (re.compile(r"(?im)^[\t ]*https?://doi\.org/10\.\d{4,9}/\S+[ \t]*$"), ""),
    (re.compile(r"(?m)^[\t ]*\d{1,4}[\t ]*$"), ""),  # page-number-only
]


_ABSTRACT_MARKER = re.compile(
    r"(?im)^(\s)*(?:Abstract[.:]?|ABSTRACT[.:]?|a\s+b\s+s\s+t\s+r\s+a\s+c\s+t)\s*$"
)


def _normalize_ligatures(text: str) -> str:
    text = text.translate(_LIGATURES)
    for a, b in _MULTI_CHAR_FIXES:
        text = text.replace(a, b)
    return text


def _skip_to_abstract(text: str, max_fraction: float = 0.33) -> str:
    """If an Abstract heading appears in the first `max_fraction` of
    text, return text from that heading onwards. Otherwise unchanged."""
    cutoff = int(len(text) * max_fraction)
    if cutoff <= 0:
        return text
    m = _ABSTRACT_MARKER.search(text[:cutoff])
    return text[m.start():] if m else text


def strip_boilerplate(text: str, *, skip_to_abstract: bool = True) -> str:
    if not text:
        return text
    text = _normalize_ligatures(text)
    for rx, repl in _BUILTIN:
        text = rx.sub(repl, text)
    if skip_to_abstract:
        text = _skip_to_abstract(text)
    return re.sub(r"\n{3,}", "\n\n", text)
