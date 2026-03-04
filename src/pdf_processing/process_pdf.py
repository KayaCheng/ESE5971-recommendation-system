'''
1.use extractor
2.use cleaner 
3.save jsonl

'''

from pathlib import Path

from src.pdf_processing.extractor import extract_pdf_pages
from src.pdf_processing.cleaner import clean_page_records
from src.pdf_processing.io_utils import save_jsonl


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_pdf = project_root / "data" / "raw" / "pdf" / "MIS1to66.pdf"
    output_jsonl = project_root / "data" / "processed" / "cleaned" / "mis1to66_pages.jsonl"

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    raw_pages = extract_pdf_pages(input_pdf)
    cleaned_pages = clean_page_records(raw_pages)
    save_jsonl(cleaned_pages, output_jsonl)

    print(f"Done. Extracted {len(cleaned_pages)} pages.")
    print(f"Saved to: {output_jsonl}")


if __name__ == "__main__":
    main()