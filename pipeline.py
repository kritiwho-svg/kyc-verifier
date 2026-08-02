"""
pipeline.py
-----------
Core KYC pipeline:
- OCR (mock/simple)
- Face detection + matching (basic)
- ELA tamper detection
- Final verdict
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageChops, ImageEnhance


# -------------------------------
# FACE DETECTION (FIXED)
# -------------------------------
def detect_face(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ✅ FIXED CASCADE PATH
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Debug check
    if face_cascade.empty():
        print("❌ Haar cascade not loaded properly")
        return []

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


# -------------------------------
# FACE MATCH (VERY BASIC)
# -------------------------------
def face_match_score(id_img, selfie_img):
    faces1 = detect_face(id_img)
    faces2 = detect_face(selfie_img)

    if len(faces1) == 0 or len(faces2) == 0:
        return 0.0  # no face detected

    # Dummy similarity (for demo)
    return 0.8  # pretend match


# -------------------------------
# ELA (Forgery Detection)
# -------------------------------
def perform_ela(image_path, output_dir):
    original = Image.open(image_path).convert("RGB")

    temp_path = os.path.join(output_dir, "temp.jpg")
    original.save(temp_path, "JPEG", quality=90)

    compressed = Image.open(temp_path)

    ela_image = ImageChops.difference(original, compressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_path = os.path.join(output_dir, "ela_output.jpg")
    ela_image.save(ela_path)

    # Overlay (simple copy for demo)
    overlay_path = os.path.join(output_dir, "overlay.jpg")
    original.save(overlay_path)

    return ela_path, overlay_path


# -------------------------------
# MOCK OCR
# -------------------------------
def extract_ocr_fields(image_path):
    return {
        "name": "John Doe",
        "id_number": "XXXX-XXXX-1234",
        "dob": "01/01/1990",
        "_ocr_confidence_hint": 0.92   # ✅ REQUIRED FIX
    }

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def run_kyc_verification(id_path, selfie_path):
    output_dir = os.path.join("outputs", "results")
    os.makedirs(output_dir, exist_ok=True)

    # OCR
    ocr_data = extract_ocr_fields(id_path)

    # Face match
    score = face_match_score(id_path, selfie_path)

    # Tamper check (ELA)
    ela_path, overlay_path = perform_ela(id_path, output_dir)

    # Verdict logic
    if score > 0.7:
        verdict = "VERIFIED"
    elif score > 0.4:
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "REJECTED"

    return {
        "ocr": ocr_data,
        "face_match": {
            "score": score
        },
        "tamper_check": {
            "ela_image_path": ela_path,
            "overlay_image_path": overlay_path
        },
        "verdict": verdict
    }