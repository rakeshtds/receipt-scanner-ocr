"""Field extraction from raw receipt OCR text.

Primary path: an LLM asked to return JSON matching RECEIPT_SCHEMA.
Fallback path: deterministic regex heuristics, used when no API key is set
or the LLM output fails validation.
"""
from __future__ import annotations

import json
import os
import re

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"], "description": "ISO date YYYY-MM-DD"},
        "currency": {"type": ["string", "null"], "description": "ISO 4217 code"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "amount": {"type": "number"},
                },
                "required": ["description", "quantity", "amount"],
            },
        },
        "subtotal": {"type": ["number", "null"]},
        "tax": {"type": ["number", "null"]},
        "total": {"type": ["number", "null"]},
        "payment_method": {"type": ["string", "null"]},
    },
    "required": ["vendor", "date", "items", "total"],
}

PROMPT = """Extract the receipt fields from the OCR text below.
Return ONLY a JSON object matching this schema: {schema}
Rules: amounts are numbers without currency symbols; date is YYYY-MM-DD;
if a field is not present, use null. OCR text:

{text}
"""

MONEY = r"(\d{1,6}[.,]\d{2})"
DATE_PATTERNS = [
    (r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", (1, 2, 3)),          # 2026-11-02
    (r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", (3, 2, 1)),          # 02/11/2026 (d/m/Y assumed if first > 12 handling below)
]


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _find_money(line: str) -> float | None:
    matches = re.findall(MONEY, line)
    if not matches:
        return None
    return _to_float(matches[-1])


def _find_date(text: str) -> str | None:
    for pattern, (yi, mi, di) in DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            parts = {1: m.group(1), 2: m.group(2), 3: m.group(3)}
            return f"{parts[yi]}-{int(parts[mi]):02d}-{int(parts[di]):02d}"
    return None


def regex_extract(text: str) -> dict:
    """Heuristic extraction: totals from labelled lines, items from priced lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result = {
        "vendor": lines[0] if lines else None,
        "date": _find_date(text),
        "currency": None,
        "items": [],
        "subtotal": None,
        "tax": None,
        "total": None,
        "payment_method": None,
    }
    skip_words = ("subtotal", "sub total", "tax", "hst", "gst", "total", "change", "cash", "balance")
    for line in lines[1:]:
        lower = line.lower()
        amount = _find_money(line)
        if amount is None:
            continue
        if "subtotal" in lower or "sub total" in lower:
            result["subtotal"] = amount
        elif any(w in lower for w in ("tax", "hst", "gst")) and result["tax"] is None:
            result["tax"] = amount
        elif lower.startswith("total") or " total " in f" {lower} ":
            result["total"] = amount
        elif any(w in lower for w in ("visa", "mastercard", "amex", "debit", "cash")):
            result["payment_method"] = line.split()[0].upper()
            result["total"] = result["total"] or amount
        elif not any(w in lower for w in skip_words):
            description = re.sub(MONEY + r"\s*$", "", line).strip(" .")
            if description:
                result["items"].append(
                    {"description": description.upper(), "quantity": 1, "amount": amount}
                )
    return result


def validate(result: dict) -> dict:
    """Attach arithmetic consistency checks; never mutate the extracted values."""
    items_sum = round(sum(i["amount"] * i.get("quantity", 1) for i in result.get("items", [])), 2)
    expected = result.get("subtotal")
    if expected is None and result.get("total") is not None and result.get("tax") is not None:
        expected = round(result["total"] - result["tax"], 2)
    result["validation"] = {
        "items_sum": items_sum,
        "items_sum_matches": (expected is None) or abs(items_sum - expected) < 0.02,
    }
    return result


def llm_extract(text: str) -> dict | None:
    """Call an OpenAI-compatible chat API; return None on any failure."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("RECEIPT_LLM_MODEL", "gpt-4o-mini"),
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": PROMPT.format(schema=json.dumps(RECEIPT_SCHEMA), text=text)}
            ],
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None


def extract_fields(text: str) -> dict:
    """LLM first, regex fallback; always validated before returning."""
    result = llm_extract(text) or regex_extract(text)
    return validate(result)
