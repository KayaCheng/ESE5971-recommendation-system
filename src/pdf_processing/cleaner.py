'''
1.clean the blank spaces
2.clean the blank rows
NEED to ADD
()


'''
import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - normalize line endings
    - strip each line
    - remove repeated blank lines
    - collapse repeated spaces
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]

    cleaned_lines = []
    prev_blank = False

    for line in lines:
        is_blank = (line == "")
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def clean_page_records(records):
    """
    Add clean_text and char_count to each extracted page record.
    """
    cleaned = []

    for record in records:
        clean = clean_text(record.get("raw_text", ""))
        new_record = dict(record)
        new_record["clean_text"] = clean
        new_record["char_count"] = len(clean)
        cleaned.append(new_record)

    return cleaned