# IITB Insti Assist - Project Write-up

## Chosen scope and why

The chosen scope was Academic Assistant as the academic pdf are properly structures and the information regarding this is standard and officially available on the internet.

Topics such as hostels or council do not have standard documents available on the internet. This also varies hostel to hostel and council to council, hence I chose Academic assistant.

## Data sources used

Located in data/raw/:

Academic_Calendar_2026-27_FINAL.pdf - everything from registration dates to date of release of grades.

Autumn_TTC_26-27_09.07.2026.pdf - slot timetable and timetable for common courses.

grading.pdf - all about the grading system

courses/* (29 files) - Department-wise running courses for Autumn 2026-27 (course code, name, instructor, slot, venue, division)

## Chunking strategy and why

**Extraction:** ingest.py extracts text from every source PDF, one record per page. The pdfs Academic Calendar, grading policy, timetable have text and are read directly with pdfplumber. The department wise course list PDFs are exported so their text is drawn as vector shapes rather than real characters, so pdfplumber finds no text on those pages. For those, the script automatically renders the page to an image and runs OCR (pytesseract) instead.

**Chunking:** build_index.py splits page text into approximately 800 character with 150 characteroverlapping chunks, embeds them with sentence-transformers (all-MiniLM-L6-v2), and stores the vectors in a FAISS index.

**Embedding + retrieval:** Chunks are embedded with `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim), a small, fast, well-established model well suited to short-document semantic search and easy to run without a GPU. The normalized vectors are stored in a FAISS `IndexFlatIP` index, so inner product is equivalent to cosine similarity — simple and exact. At query time the top-5 chunks are retrieved (configurable in the UI) and pass them to Claude with an explicit instruction to answer only from that context and say "I don't know"
otherwise, plus a rule against stating specific facts (dates, grades, slots) that aren't grounded in the retrieved text. The UI displays which documents and pages were used for every answer.

## Known limitations / what I'd improve with more time

- **OCR noise on course tables.** The department course-list PDFs are dense, merged-cell tables (course code, name, instructor, venue, slot, division all in one row). OCR handles the running text well but sometimes misreads or mangles adjacent table cells, especially long multi-line slot/timing entries. This makes the assistant reliable for "what courses does Department X offer" but less reliable for pulling an exact room number or slot code out of a specific row. With more time I'd parse these as structured tables instead of chunking raw OCR text.

- **Length-based, not table-aware, chunking.** A course row can be split across two chunks if it's long, separating a course code from its instructor. Row-aware or table-aware chunking would fix this.