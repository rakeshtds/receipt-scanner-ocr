"""Receipt Scanner API.

POST /scan with a receipt image -> structured JSON fields.
Run: uvicorn app.main:app --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

try:
    from .extract import extract_fields
    from .ocr import run_ocr
except ImportError:  # running as a loose script from the app/ folder
    from extract import extract_fields
    from ocr import run_ocr

app = FastAPI(title="Receipt Scanner OCR", version="1.0.0")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
MAX_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/scan")
async def scan(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")
    text = run_ocr(data)
    if not text:
        raise HTTPException(status_code=422, detail="OCR produced no text; try a sharper photo")
    fields = extract_fields(text)
    fields["raw_text"] = text
    return fields
