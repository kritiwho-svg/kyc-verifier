from ocr_extractor import extract_text
from tamper_detector import analyze_tampering
from face_match_deepface import compare_faces


def run_kyc(id_image, selfie_image):
    print("\n===== KYC VERIFICATION START =====\n")

    # 1. OCR
    print("🔍 Running OCR...")
    try:
        text = extract_text(id_image)
        print("OCR Output:\n", text)
    except Exception as e:
        print("OCR failed:", e)

    # 2. Tamper Detection
    print("\n🛡️ Checking Tampering...")
    tamper_result = analyze_tampering(id_image)
    print("Tamper Score:", tamper_result["tamper_score"])
    print("Verdict:", tamper_result["verdict"])

    # 3. Face Matching
    print("\n🙂 Comparing Faces...")
    face_result = compare_faces(id_image, selfie_image)
    print("Face Match:", face_result)

    print("\n===== FINAL RESULT =====")
    if tamper_result["verdict"] == "LIKELY_TAMPERED":
        print("❌ Document is tampered")
    elif face_result.get("match"):
        print("✅ KYC VERIFIED SUCCESSFULLY")
    else:
        print("❌ Face mismatch")



if __name__ == "__main__":
    run_kyc(
        "sample_data/mock_aadhaar_genuine.jpg",
        "sample_data/selfie_match.jpg"
    )