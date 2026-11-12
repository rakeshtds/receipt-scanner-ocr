"""Image preprocessing and Tesseract OCR for receipt photos."""
from __future__ import annotations

import io

import pytesseract
from PIL import Image, ImageOps

# Receipt-tuned Tesseract config: assume a single uniform block of text,
# keep whitespace so column layout (item .... price) survives extraction.
TESSERACT_CONFIG = "--psm 6 -c preserve_interword_spaces=1"

MAX_DIMENSION = 2000  # px; larger images are downscaled for speed


def preprocess(image_bytes: bytes) -> Image.Image:
    """Load an image, normalize orientation, grayscale and upscale small text."""
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    if image.mode != "L":
        image = image.convert("L")
    # Upscale small receipts: Tesseract reads small print poorly.
    width, height = image.size
    longest = max(width, height)
    if longest < 1200:
        scale = 1200 / longest
        image = image.resize((int(width * scale), int(height * scale)))
    elif longest > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest
        image = image.resize((int(width * scale), int(height * scale)))
    # Mild contrast boost helps faded thermal-print receipts.
    image = ImageOps.autocontrast(image, cutoff=1)
    return image


def run_ocr(image_bytes: bytes) -> str:
    """Return the raw OCR text of a receipt image."""
    image = preprocess(image_bytes)
    text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)
    return text.strip()
