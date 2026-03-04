'''
1.clean the blank spaces
2.clean the blank rows
NEED to ADD
()


'''
import re


def replace_ligatures(text: str) -> str:
    """
    Replace common PDF ligature characters with normal ASCII sequences.
    """
    replacements = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_standalone_page_numbers(lines):
    """
    Remove lines that are just page numbers, e.g. '10'.
    """
    cleaned = []
    for line in lines:
        if re.fullmatch(r"\d+", line.strip()):
            continue
        cleaned.append(line)
    return cleaned


def merge_broken_hyphenation(text: str) -> str:
    """
    Fix line-break hyphenation:
    'state-of-the-\\nart' -> 'state-of-the-art'
    'X-\\nrays' -> 'X-rays'
    """
    return re.sub(r"-\n(?=\w)", "-", text)


def merge_wrapped_lines(text: str) -> str:
    """
    Merge wrapped lines inside paragraphs while preserving paragraph breaks.
    Strategy:
    - keep empty lines as paragraph boundaries
    - join consecutive non-empty lines with spaces
    """
    lines = [line.strip() for line in text.split("\n")]

    paragraphs = []
    buffer = []

    for line in lines:
        if line == "":
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
        else:
            buffer.append(line)

    if buffer:
        paragraphs.append(" ".join(buffer))

    return "\n\n".join(paragraphs)


def clean_text(text: str) -> str:
    """
    Stronger text cleaning for PDF-extracted book pages.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = replace_ligatures(text)
    text = merge_broken_hyphenation(text)

    lines = [line.strip() for line in text.split("\n")]
    lines = remove_standalone_page_numbers(lines)

    # remove repeated blank lines
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        is_blank = (line == "")
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank

    text = "\n".join(cleaned_lines)

    # collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text).strip()

    # merge wrapped body lines into paragraph-like text
    text = merge_wrapped_lines(text)

    return text.strip()


def filter_body_pages(records, start_page=11):
    """
    Keep only pages starting from the main body.
    """
    return [r for r in records if r.get("page_num", 0) >= start_page]


def clean_page_records(records):
    """
    Filter body pages and add cleaned text.
    """
    body_records = filter_body_pages(records, start_page=11)

    cleaned = []
    for record in body_records:
        clean = clean_text(record.get("raw_text", ""))
        new_record = dict(record)
        new_record["clean_text"] = clean
        new_record["char_count"] = len(clean)
        cleaned.append(new_record)

    return cleaned