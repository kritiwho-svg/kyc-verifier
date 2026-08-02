"""
decision.py
Combines face match score, OCR confidence, and forgery score into
a single final verdict: VERIFIED / REJECTED / NEEDS_REVIEW.

This is a simple rule-based engine (not ML) so it's transparent and
easy to tune. Thresholds below are starting points -- adjust based
on testing with real sample data.
"""

# --- Thresholds (tune these after testing) ---
FACE_MATCH_VERIFIED_MIN = 70.0     # similarity_score >= this -> face OK
FACE_MATCH_REVIEW_MIN = 50.0       # between REVIEW_MIN and VERIFIED_MIN -> review

OCR_CONFIDENCE_MIN = 0.4           # avg_confidence below this -> unreliable OCR

FORGERY_REJECTED_MIN = 60.0        # forgery_score >= this -> likely tampered
FORGERY_REVIEW_MIN = 35.0          # between REVIEW_MIN and REJECTED_MIN -> review


def get_verdict(face_result: dict, ocr_result: dict, forgery_result: dict) -> dict:
    """
    Inputs are the dicts returned by:
        face_match.get_face_match_score()
        ocr.extract_fields()
        forgery.get_forgery_score()

    Returns:
        {
            "verdict": "VERIFIED" | "REJECTED" | "NEEDS_REVIEW",
            "reasons": [list of strings explaining the verdict]
        }
    """
    reasons = []

    similarity = face_result.get("similarity_score", 0.0)
    ocr_conf = ocr_result.get("avg_confidence", 0.0)
    forgery_score = forgery_result.get("forgery_score", 100.0)  # fail-safe: assume worst if missing

    # --- Hard rejects first ---
    if forgery_score >= FORGERY_REJECTED_MIN:
        reasons.append(f"High forgery score ({forgery_score}) suggests tampering.")
        return {"verdict": "REJECTED", "reasons": reasons}

    if similarity < FACE_MATCH_REVIEW_MIN:
        reasons.append(f"Face similarity too low ({similarity}).")
        return {"verdict": "REJECTED", "reasons": reasons}

    if ocr_result.get("name") == "NOT_FOUND" and ocr_result.get("id_number") == "NOT_FOUND":
        reasons.append("Could not extract any identifying fields from ID.")
        return {"verdict": "REJECTED", "reasons": reasons}

    # --- Review zone ---
    needs_review = False

    if FACE_MATCH_REVIEW_MIN <= similarity < FACE_MATCH_VERIFIED_MIN:
        reasons.append(f"Face similarity borderline ({similarity}).")
        needs_review = True

    if FORGERY_REVIEW_MIN <= forgery_score < FORGERY_REJECTED_MIN:
        reasons.append(f"Forgery score borderline ({forgery_score}).")
        needs_review = True

    if ocr_conf < OCR_CONFIDENCE_MIN:
        reasons.append(f"Low OCR confidence ({ocr_conf}).")
        needs_review = True

    if needs_review:
        return {"verdict": "NEEDS_REVIEW", "reasons": reasons}

    # --- All checks passed ---
    reasons.append("Face match strong, OCR confident, no signs of tampering.")
    return {"verdict": "VERIFIED", "reasons": reasons}


if __name__ == "__main__":
    # Quick manual test with fake sample data
    fake_face = {"similarity_score": 85.0}
    fake_ocr = {"avg_confidence": 0.8, "name": "JOHN DOE", "id_number": "1234 5678 9012"}
    fake_forgery = {"forgery_score": 10.0}

    result = get_verdict(fake_face, fake_ocr, fake_forgery)
    print("\n--- DECISION RESULT (sample data) ---")
    print(result)
