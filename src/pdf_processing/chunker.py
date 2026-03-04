from typing import List, Dict, Any


def is_likely_title(paragraph: str) -> bool:
    """
    Heuristic: detect whether a paragraph looks like a title/header.
    This helps preserve chapter/section headings as chunk boundaries.
    """
    if not paragraph:
        return False

    text = paragraph.strip()

    title_keywords = [
        "chapter",
        "introduction",
        "system theory",
        "image processing",
        "signals and systems",
        "fourier transform",
        "convolution",
        "correlation",
        "morphological operators",
    ]

    lower = text.lower()

    if any(k in lower for k in title_keywords):
        return True

    # short-ish line, no ending punctuation, likely a heading
    if len(text) < 80 and not text.endswith((".", "!", "?", ";", ":")):
        # if it has only a few words, it may be a title
        if len(text.split()) <= 8:
            return True

    return False


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split cleaned page text into paragraph-like units.
    Assumes paragraphs are separated by blank lines.
    """
    if not text:
        return []

    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def should_skip_page(record: Dict[str, Any], min_chars: int = 80) -> bool:
    """
    Skip empty or extremely short low-information pages.
    Preserve short title-like pages if they look useful.
    """
    text = record.get("clean_text", "").strip()
    char_count = record.get("char_count", 0)

    if not text:
        return True

    if char_count >= min_chars:
        return False

    # keep short title-like pages
    if is_likely_title(text):
        return False

    return True


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate for English text.
    """
    return max(1, len(text) // 4)


def finalize_chunk(
    chunk_id: int,
    source_name: str,
    text_parts: List[str],
    page_start: int,
    page_end: int,
) -> Dict[str, Any]:
    """
    Build one chunk record.
    """
    chunk_text = "\n\n".join(part.strip() for part in text_parts if part.strip()).strip()
    char_count = len(chunk_text)

    return {
        "chunk_id": f"mis1to66_chunk_{chunk_id:04d}",
        "source_name": source_name,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_text": chunk_text,
        "char_count": char_count,
        "token_estimate": estimate_tokens(chunk_text),
        "prev_chunk_id": None,   # fill later
        "next_chunk_id": None,   # fill later
    }


def link_chunk_neighbors(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fill prev_chunk_id and next_chunk_id after all chunks are created.
    """
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk["prev_chunk_id"] = chunks[i - 1]["chunk_id"]
        if i < len(chunks) - 1:
            chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"]
    return chunks


def build_chunks_from_pages(
    page_records: List[Dict[str, Any]],
    target_chars: int = 1200,
    min_chars: int = 600,
    max_chars: int = 1800,
) -> List[Dict[str, Any]]:
    """
    Build chunk records from cleaned page-level records.

    Strategy:
    - iterate pages in order
    - split each page into paragraph-like units
    - accumulate paragraphs into a chunk
    - flush chunk when:
        * adding a paragraph would exceed max_chars, or
        * current chunk has reached target_chars and next paragraph is a title
    """
    if not page_records:
        return []

    # sort pages just in case
    page_records = sorted(page_records, key=lambda x: x.get("page_num", 0))

    source_name = page_records[0].get("source_name", "unknown.pdf")
    chunks = []

    current_parts: List[str] = []
    current_len = 0
    current_page_start = None
    current_page_end = None
    chunk_counter = 1

    for record in page_records:
        if should_skip_page(record):
            continue

        page_num = record.get("page_num")
        text = record.get("clean_text", "")
        paragraphs = split_into_paragraphs(text)

        if not paragraphs:
            continue

        for para in paragraphs:
            para_len = len(para)

            # if starting a new chunk
            if not current_parts:
                current_parts = [para]
                current_len = para_len
                current_page_start = page_num
                current_page_end = page_num
                continue

            # if next paragraph is a likely title and current chunk is already big enough,
            # close current chunk first so title starts a new chunk
            if is_likely_title(para) and current_len >= min_chars:
                chunks.append(
                    finalize_chunk(
                        chunk_counter,
                        source_name,
                        current_parts,
                        current_page_start,
                        current_page_end,
                    )
                )
                chunk_counter += 1

                current_parts = [para]
                current_len = para_len
                current_page_start = page_num
                current_page_end = page_num
                continue

            # if adding this paragraph would exceed max_chars, flush current chunk first
            if current_len + 2 + para_len > max_chars and current_len >= min_chars:
                chunks.append(
                    finalize_chunk(
                        chunk_counter,
                        source_name,
                        current_parts,
                        current_page_start,
                        current_page_end,
                    )
                )
                chunk_counter += 1

                current_parts = [para]
                current_len = para_len
                current_page_start = page_num
                current_page_end = page_num
                continue

            # otherwise keep accumulating
            current_parts.append(para)
            current_len += 2 + para_len
            current_page_end = page_num

            # optional flush if we've reached target and current paragraph ends naturally
            if current_len >= target_chars and para.endswith((".", "!", "?", ";")):
                chunks.append(
                    finalize_chunk(
                        chunk_counter,
                        source_name,
                        current_parts,
                        current_page_start,
                        current_page_end,
                    )
                )
                chunk_counter += 1

                current_parts = []
                current_len = 0
                current_page_start = None
                current_page_end = None

    # flush remainder
    if current_parts:
        chunks.append(
            finalize_chunk(
                chunk_counter,
                source_name,
                current_parts,
                current_page_start,
                current_page_end,
            )
        )

    return link_chunk_neighbors(chunks)