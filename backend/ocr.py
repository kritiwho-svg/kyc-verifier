"""
ocr.py
Extracts raw text from an ID card image using EasyOCR, then parses
Name / DOB / ID Number using simple regex heuristics.

NOTE: EasyOCR downloads its detection/recognition model weights the
first time it runs (~50-100MB). Needs internet on first run only.
"""

import re
import easyocr
import numpy as np
import cv2

# Reader is created once and reused (loading it per-call is slow).
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        # 'en' only for now; add more langs later if needed e.g. ['en', 'hi']
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text(image_path: str) -> list:
    """
    Runs OCR on the given image path.
    Returns a list of (text, confidence) tuples.
    """
    reader = _get_reader()
    results = reader.readtext(image_path)  # [(bbox, text, conf), ...]

    extracted = []
    for _, text, conf in results:
        clean = text.strip()
        if clean:
            extracted.append((clean, float(conf)))
    return extracted


def _find_dob(lines: list) -> str:
    """Looks for common date formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY"""
    date_pattern = re.compile(r"(\d{2}[/\-.]\d{2}[/\-.]\d{4})")
    for text, _ in lines:
        match = date_pattern.search(text)
        if match:
            return match.group(1)
    return "NOT_FOUND"


def _find_id_number(lines: list) -> str:
    """
    Looks for common ID patterns:
    - Aadhaar-style: 4-4-4 digit groups (mock only, no real PII)
    - PAN-style: 5 letters + 4 digits + 1 letter
    """
    aadhaar_pattern = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
    pan_pattern = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

    for text, _ in lines:
        upper = text.upper().replace(" ", "")
        pan_match = pan_pattern.search(upper)
        if pan_match:
            return pan_match.group(0)

    for text, _ in lines:
        aadhaar_match = aadhaar_pattern.search(text)
        if aadhaar_match:
            return aadhaar_match.group(0)

    return "NOT_FOUND"


def _find_name(lines: list) -> str:
    """
    Heuristic: name is usually a line with only alphabetic characters
    and spaces, at least 2 words, not containing common ID-card keywords.
    This is a heuristic, not a perfect parser -- real deployments need
    layout-aware parsing per document type.
    """
    keywords_to_skip = {
        "government", "india", "male", "female", "dob", "date", "birth",
        "income", "tax", "department", "permanent", "account", "number",
        "signature", "address", "aadhaar", "unique", "identification"
    }

    candidates = []
    for text, conf in lines:
        cleaned = text.strip()
        if not cleaned:
            continue
        letters_only = re.sub(r"[^A-Za-z ]", "", cleaned)
        if len(letters_only) < 4:
            continue
        word_count = len(letters_only.split())
        if word_count < 2:
            continue
        lower = letters_only.lower()
        if any(k in lower for k in keywords_to_skip):
            continue
        candidates.append((letters_only.strip(), conf))

    if not candidates:
        return "NOT_FOUND"

    # Pick the highest-confidence candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def extract_fields(image_path: str) -> dict:
    """
    Main entry point: runs OCR and extracts structured fields.
    Returns dict with name, dob, id_number, raw_text, avg_confidence.
    """
    lines = extract_text(image_path)

    if not lines:
        return {
            "name": "NOT_FOUND",
            "dob": "NOT_FOUND",
            "id_number": "NOT_FOUND",
            "raw_text": "",
            "avg_confidence": 0.0,
        }

    avg_conf = sum(c for _, c in lines) / len(lines)

    return {
        "name": _find_name(lines),
        "dob": _find_dob(lines),
        "id_number": _find_id_number(lines),
        "raw_text": " | ".join(t for t, _ in lines),
        "avg_confidence": round(avg_conf, 3),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr.py <path_to_id_image>")
        sys.exit(1)

    path = sys.argv[1]
    result = extract_fields(path)
    print("\n--- OCR RESULT ---")
    for k, v in result.items():
        print(f"{k}: {v}")
