# Receipt Scanner OCR

Snap a photo of a receipt and get clean, structured JSON — vendor, date, line items, taxes and totals — via Tesseract OCR plus LLM field extraction with a regex fallback.

## Features

- Image preprocessing (grayscale, contrast, deskew-friendly resizing) before OCR
- Tesseract OCR with a receipt-tuned config
- LLM extraction against a strict JSON schema (vendor, date, currency, line items, subtotal, tax, total, payment method)
- Pure-regex fallback extractor when no LLM API key is configured, so the service always returns something useful
- Validation pass that reconciles line-item sums against the printed total and flags mismatches
- FastAPI service with a single `/scan` endpoint plus a `/health` check

## Architecture

`POST /scan` accepts an image upload → `app/ocr.py` runs preprocessing + Tesseract and returns raw text → `app/extract.py` sends the text to an LLM with a JSON-schema prompt (or runs the regex fallback) → a validator checks arithmetic consistency → structured JSON is returned.

## Tech Stack

Python 3.11 · FastAPI · pytesseract · Pillow · OpenAI-compatible LLM API · Uvicorn

## Setup

```bash
# system dependency: Tesseract OCR engine
sudo apt install tesseract-ocr        # Debian/Ubuntu
brew install tesseract                # macOS

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...          # optional; regex fallback works without it
```

## Usage

```bash
uvicorn app.main:app --reload --port 8000
curl -X POST http://localhost:8000/scan -F "file=@receipt.jpg"
```

## Example Output

See `sample/receipt_fields.json` for the exact response shape:

```json
{
  "vendor": "MAPLE LEAF COFFEE CO.",
  "date": "2026-11-02",
  "currency": "CAD",
  "items": [{"description": "FLAT WHITE", "quantity": 1, "amount": 4.75}],
  "subtotal": 12.25, "tax": 1.59, "total": 13.84,
  "payment_method": "VISA",
  "validation": {"items_sum_matches": true}
}
```

## Project Structure

```
app/main.py       FastAPI app and /scan endpoint
app/ocr.py        Image preprocessing + Tesseract OCR
app/extract.py    LLM extraction (JSON schema) + regex fallback + validation
sample/           Example structured output
```

## Roadmap

- Batch scanning and CSV export for expense reports
- Multi-language receipt support (Tesseract language packs)
- Confidence scoring per field with human-in-the-loop review queue

---

An original implementation built for portfolio purposes. Sample data is synthetic.
