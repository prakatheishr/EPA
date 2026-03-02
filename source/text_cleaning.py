import re

# Remove label leakage patterns from the text
LABEL_RE = re.compile(
    r"""
    # Explicit MES / Mayo patterns (all 0–3)
    (\bmes\s*[-]?\s*[0-3]\b) |
    (\bmayo\s*endoscopic\s*score\s*[0-3]\b) |
    (\bmayo\s*score\s*[0-3]\b) |

    # "support(s) a 0/1/2/3" or "support(s) mes1"
    (\bsupports?\s+(a\s+)?(mes\s*)?[0-3]\b) |
    (\bsupports?\s+(a\s+)?mes\s*[0-3]\b) |

    # "consistent with 0/1/2/3" or "consistent with mes2"
    (\bconsistent\s+with\s+(mes\s*)?[0-3]\b) |

    # "grade 0/1/2/3" or "grade of 0/1/2/3"
    (\bgrade(\s+of)?\s+(mes\s*)?[0-3]\b)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

def clean_caption_body(text: str) -> str:
    # remove obvious MES label cues from the description body text.
    text = "" if text is None else str(text)
    text = LABEL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def format_caption_with_mes(clean_body: str, mes_num: int) -> str:
    # append the MES grade in the exact style you want.
    clean_body = (clean_body or "").strip()
    suffix = f" these findings support a mes{int(mes_num)}"
    return (clean_body + suffix).strip().lower()