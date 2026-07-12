import json
import os
import sys
from pathlib import Path

import pdfplumber
from pdf2image import convert_from_path
import pytesseract

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "pages.jsonl"

MIN_TEXT_LEN = 20


def extract_with_pdfplumber(pdf_path: Path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text.strip()))
    return pages


def extract_with_ocr(pdf_path: Path, page_numbers):
    results = {}
    for page_num in page_numbers:
        images = convert_from_path(
            str(pdf_path), dpi=200, first_page=page_num, last_page=page_num
        )
        text = pytesseract.image_to_string(images[0], config="--psm 6")
        results[page_num] = text.strip()
    return results


def process_pdf(pdf_path: Path, relative_name: str):
    records = []
    pages = extract_with_pdfplumber(pdf_path)

    needs_ocr = [num for num, text in pages if len(text) < MIN_TEXT_LEN]
    ocr_results = {}
    if needs_ocr:
        print(f"  OCR needed for {relative_name}: pages {needs_ocr}")
        ocr_results = extract_with_ocr(pdf_path, needs_ocr)

    for page_num, text in pages:
        final_text = ocr_results.get(page_num, text)
        if final_text:
            records.append({"source": relative_name, "page": page_num, "text": final_text})
    return records


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(RAW_DIR.glob("*.pdf")) + sorted(RAW_DIR.glob("courses/*.pdf"))

    # Optional CLI args let this be run in batches: `python ingest.py START END`
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(pdf_paths)
    mode = "a" if start > 0 else "w"
    pdf_paths = pdf_paths[start:end]

    total = 0
    with open(OUT_PATH, mode, encoding="utf-8") as f:
        for pdf_path in pdf_paths:
            relative_name = str(pdf_path.relative_to(RAW_DIR))
            print(f"Processing {relative_name} ...", flush=True)
            records = process_pdf(pdf_path, relative_name)
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            total += len(records)

    print(f"\nDone batch [{start}:{end}]. Extracted {total} pages from {len(pdf_paths)} PDFs.")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
