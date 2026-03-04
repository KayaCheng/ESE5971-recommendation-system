'''
1.use extractor
2.use cleaner 
3.save jsonl

'''

from pathlib import Path

from src.pdf_processing.extractor import extract_pdf_pages
from src.pdf_processing.cleaner import clean_page_records
from src.pdf_processing.chunker import build_chunks_from_pages
from src.pdf_processing.io_utils import save_jsonl


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_pdf = project_root / "data" / "raw" / "pdf" / "MIS1to66.pdf"

    cleaned_output = project_root / "data" / "processed" / "cleaned" / "mis1to66_pages.jsonl"
    chunks_output = project_root / "data" / "processed" / "chunks" / "mis1to66_chunks.jsonl"

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    # Step 1: extract raw page text
    raw_pages = extract_pdf_pages(input_pdf)

    # Step 2: clean and keep body pages only
    cleaned_pages = clean_page_records(raw_pages)
    save_jsonl(cleaned_pages, cleaned_output)

    # Step 3: build chunks
    chunks = build_chunks_from_pages(cleaned_pages)
    save_jsonl(chunks, chunks_output)

    print(f"Done. Extracted {len(raw_pages)} raw pages.")
    print(f"Kept {len(cleaned_pages)} cleaned body pages.")
    print(f"Built {len(chunks)} chunks.")
    print(f"Saved cleaned pages to: {cleaned_output}")
    print(f"Saved chunks to: {chunks_output}")


if __name__ == "__main__":
    main()