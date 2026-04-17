import re

# remove boilerplate phrases that appear across many reports
BOILERPLATE_RE = re.compile(
    r"""
    (these\s+findings\s+support(\s+an?)?)|
    (findings\s+support(\s+an?)?)|
    (overall\s+findings\s+are\s+consistent\s+with)|
    (consistent\s+with)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# remove MES mentions in any format (mes2, mes-2, mes 2, mayo endoscopic score 2 etc.)
MES_RE = re.compile(
    r"""
    (\bmes\s*[-]?\s*[0-3]\b)|
    (\bmayo\s*endoscopic\s*score\s*[0-3]\b)|
    (\bmayo\s*score\s*[0-3]\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

def clean_caption_body(text: str) -> str:
    text = "" if text is None else str(text).strip()
    text = MES_RE.sub("", text)
    text = BOILERPLATE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" .,-;:")
    return text

def format_caption_with_mes(body: str, mes_num: int) -> str:
    body = body.strip()
    prefix = "Findings: "
    suffix = f" MES-{mes_num}."
    # GPT-2 EOS is tokenizer.eos_token - appended via tokenizer
    eos = ""  
    if not body:
        return prefix + suffix.strip()
    return prefix + body + suffix + eos