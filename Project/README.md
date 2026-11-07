# IITB Insti Assist - Academic Assistant

A RAG assistant which answers questions related to academics of IITB, courses offered, grading, academic calender, timetable for Autumn semester 2026-27.

## How it works

PDFs regarding the topics mentioned above were downloaded from the academic website. For the running courses, pdf printouts were taken from asc portal. These pdfs are screenshots of the running courses webpage of each department.

1. **ingest.py** - it extracts text from every source PDF, one record per page. The pdfs Academic Calendar, grading policy, timetable have text and are read directly with pdfplumber. The department wise course list PDFs are exported so their text is drawn as vector shapes rather than real characters, so pdfplumber finds no text on those pages. For those, the script automatically renders the page to an image and runs OCR (pytesseract) instead.

2. **build_index.py** - it splits page text into approximately 800 character overlapping chunks, embeds them with sentence-transformers (all-MiniLM-L6-v2), and stores the vectors in a FAISS index.

3. **rag.py** - it takes questions, embeds them with the same model, retrieves the top-k most similar chunks from FAISS, and sends them to Google Gemini as context.

4. **app.py** — it a small Streamlit chat UI that shows the answer and sources. 

## Data sources

Located in data/raw/:

Academic_Calendar_2026-27_FINAL.pdf - everything from registration dates to date of release of grades.

Autumn_TTC_26-27_09.07.2026.pdf - slot timetable and timetable for common courses.

grading.pdf - all about the grading system

courses/* (29 files) - Department-wise running courses for Autumn 2026-27 (course code, name, instructor, slot, venue, division)

## Setup

### 1. System dependencies (needed for OCR on the course-list PDFs)

```bash
# Debian/Ubuntu
sudo apt-get install poppler-utils tesseract-ocr
```

### 2. Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. API key

Create a free token at [Gemini AI Studio](https://aistudio.google.com/api-keys) and set it:

```bash
export GEMINI_API_KEY=your-token-here
```

## Running the pipeline

data/processed/pages.jsonl (the OCR'd/extracted text) is already included in this repo, so you can skip straight to building the index. Re-run ingest.py only if you add/replace PDFs in data/raw/.

```bash
python src/ingest.py          # optional: PDFs -> data/processed/pages.jsonl  (~5-10 min, OCR is the slow part)
python src/build_index.py     # required: pages.jsonl -> data/index/ (downloads the embedding model on first run)
```

Then launch the app:

```bash
streamlit run src/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and start asking questions.