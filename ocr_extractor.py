"""
ocr_extractor.py
-----------------
Runs OCR over a mock ID card image (Aadhaar/PAN style) and parses out structured
fields using regex heuristics. Uses Tesseract via pytesseract -- no cloud calls,
runs fully offline.
"""

import re
import cv2
import pytesseract
import numpy as np

AADHAAR_RE = re.compile(r"\b(\d{4}\s?\d{4}\s?\d{4})\b")
PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
DOB_RE = re.compile(r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b")
NAME_RE = re.compile(r"Name[:\-]?\s*([A-Z][A-Za-z .]{2,40})")
FATHER_RE = re.compile(r"Father'?s?\s*Name[:\-]?\s*([A-Z][A-Za-z .]{2,40})")
GENDER_RE = re.compile(r"Gender[:\-]?\s*(Male|Female|Other|M|F)", re.IGNORECASE)


def _preprocess(image_path):
    """Light preprocessing to improve OCR accuracy: grayscale + adaptive threshold."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return img, thresh


def extract_text(image_path):
    """Returns raw OCR text plus per-word bounding-box data from the ID image."""
    _, processed = _preprocess(image_path)
    raw_text = pytesseract.image_to_string(processed)
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    return raw_text, data


def parse_fields(raw_text):
    """Applies regex heuristics over the OCR text to pull out structured KYC fields.
    Detects document type automatically based on which ID pattern is present."""
    fields = {
        "doc_type": None,
        "name": None,
        "father_name": None,
        "dob": None,
        "gender": None,
        "id_number": None,
    }

    aadhaar_match = AADHAAR_RE.search(raw_text)
    pan_match = PAN_RE.search(raw_text)

    if pan_match:
        fields["doc_type"] = "PAN"
        fields["id_number"] = pan_match.group(1)
    elif aadhaar_match:
        fields["doc_type"] = "AADHAAR"
        fields["id_number"] = aadhaar_match.group(1).replace(" ", " ").strip()

    name_match = NAME_RE.search(raw_text)
    if name_match:
        fields["name"] = name_match.group(1).strip()

    father_match = FATHER_RE.search(raw_text)
    if father_match:
        fields["father_name"] = father_match.group(1).strip()

    dob_match = DOB_RE.search(raw_text)
    if dob_match:
        fields["dob"] = dob_match.group(1)

    gender_match = GENDER_RE.search(raw_text)
    if gender_match:
        fields["gender"] = gender_match.group(1).capitalize()

    return fields


def extract_id_fields(image_path):
    """Full pipeline step: OCR + parse. Returns (fields_dict, raw_text)."""
    raw_text, _ = extract_text(image_path)
    fields = parse_fields(raw_text)
    fields["_ocr_confidence_hint"] = _rough_confidence(raw_text, fields)
    return fields, raw_text


def _rough_confidence(raw_text, fields):
    """Cheap heuristic confidence score: fraction of expected fields successfully parsed."""
    expected = ["doc_type", "name", "dob", "id_number"]
    found = sum(1 for k in expected if fields.get(k))
    return round(found / len(expected), 2)


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/mock_aadhaar_genuine.jpg"
    fields, raw = extract_id_fields(path)
    print("---- RAW OCR TEXT ----")
    print(raw)
    print("---- PARSED FIELDS ----")
    print(json.dumps(fields, indent=2))
