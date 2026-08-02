"""
main.py
Single entry point that ties together OCR, face matching, and forgery
detection into one verify_kyc() function returning a JSON-serializable dict.
"""

import os
import json
import traceback

from backend.ocr import extract_fields
from backend.face_match import get_face_match_score, FaceMatchError
from backend.forgery import get_forgery_score
from backend.decision import get_verdict


def verify_kyc(id_image_path: str, selfie_image_path: str, ela_output_path: str = None) -> dict:
    """
    Runs the full KYC pipeline on a given ID image and selfie image.

    Args:
        id_image_path: path to the ID card image (mock Aadhaar/PAN, no real PII)
        selfie_image_path: path to the selfie image
        ela_output_path: optional path to save the ELA visualization image

    Returns:
        dict with keys: ocr, face_match, forgery, decision, errors
    """
    result = {
        "ocr": None,
        "face_match": None,
        "forgery": None,
        "decision": None,
        "errors": [],
    }

    # --- OCR ---
    try:
        result["ocr"] = extract_fields(id_image_path)
    except Exception as e:
        result["errors"].append(f"OCR failed: {e}")
        result["ocr"] = {
            "name": "NOT_FOUND", "dob": "NOT_FOUND",
            "id_number": "NOT_FOUND", "raw_text": "", "avg_confidence": 0.0,
        }

    # --- Face match ---
    try:
        result["face_match"] = get_face_match_score(id_image_path, selfie_image_path)
    except FaceMatchError as e:
        result["errors"].append(f"Face match failed: {e}")
        result["face_match"] = {
            "verified": False, "distance": None,
            "similarity_score": 0.0, "model": "Facenet512",
        }
    except Exception as e:
        result["errors"].append(f"Face match unexpected error: {e}")
        result["face_match"] = {
            "verified": False, "distance": None,
            "similarity_score": 0.0, "model": "Facenet512",
        }

    # --- Forgery / ELA ---
    try:
        result["forgery"] = get_forgery_score(id_image_path, output_path=ela_output_path)
    except Exception as e:
        result["errors"].append(f"Forgery check failed: {e}")
        result["forgery"] = {
            "forgery_score": 100.0, "ela_image_path": None,
            "mean_ela_intensity": 0.0, "max_ela_intensity": 0.0,
        }

    # --- Decision ---
    result["decision"] = get_verdict(result["face_match"], result["ocr"], result["forgery"])

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m backend.main <id_image_path> <selfie_image_path>")
        sys.exit(1)

    id_path, selfie_path = sys.argv[1], sys.argv[2]

    try:
        output = verify_kyc(id_path, selfie_path)
        print(json.dumps(output, indent=2))
    except Exception:
        print("FATAL ERROR:")
        traceback.print_exc()
