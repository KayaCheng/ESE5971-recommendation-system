'''
From pdf to raw text
1.open pdf
2.extract text according to pages
3.return page-level records

'''
from pathlib import Path
import fitz  # PyMuPDF


def extract_pdf_pages(pdf_path: Path):
    """
    Extract raw text page by page from a PDF.
    Returns a list of page-level records.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        raw_text = page.get_text("text")
        pages.append(
            {
                "source_name": pdf_path.name,
                "page_num": i + 1,
                "raw_text": raw_text,
            }
        )

    doc.close()
    return pages