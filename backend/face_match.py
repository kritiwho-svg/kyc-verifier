"""
face_match.py
Compares the face on the ID card against the selfie using DeepFace
(embedding-based similarity, model: Facenet512).

IMPORTANT: The env var below must be set BEFORE deepface is imported
anywhere in the process, otherwise TensorFlow 2.16 (Keras 3) conflicts
with DeepFace's legacy Keras 2 model code.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace


class FaceMatchError(Exception):
    """Raised when a face cannot be detected in one or both images."""
    pass


def get_face_match_score(id_image_path: str, selfie_image_path: str) -> dict:
    """
    Returns a dict:
        {
            "verified": bool,          # DeepFace's own threshold-based verdict
            "distance": float,         # raw distance (lower = more similar)
            "similarity_score": float, # 0-100 scale, higher = more similar
            "model": str
        }
    Raises FaceMatchError if no face is detected in either image.
    """
    model_name = "Facenet512"

    try:
        result = DeepFace.verify(
            img1_path=id_image_path,
            img2_path=selfie_image_path,
            model_name=model_name,
            detector_backend="opencv",   # fast, no extra install needed
            enforce_detection=True,      # raises if no face found
        )
    except ValueError as e:
        # DeepFace raises ValueError with "Face could not be detected"
        # when enforce_detection=True and no face is found.
        raise FaceMatchError(
            f"Face detection failed on one of the images. Details: {e}"
        )

    distance = float(result["distance"])
    threshold = float(result["threshold"])

    # Convert distance -> a 0-100 similarity score.
    # DeepFace distance: 0 = identical, higher = less similar.
    # We normalize using the threshold as the "50% similarity" anchor.
    similarity_score = max(0.0, min(100.0, (1 - (distance / (threshold * 2))) * 100))

    return {
        "verified": bool(result["verified"]),
        "distance": round(distance, 4),
        "threshold": round(threshold, 4),
        "similarity_score": round(similarity_score, 2),
        "model": model_name,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python face_match.py <id_image> <selfie_image>")
        sys.exit(1)

    id_path, selfie_path = sys.argv[1], sys.argv[2]

    try:
        match_result = get_face_match_score(id_path, selfie_path)
        print("\n--- FACE MATCH RESULT ---")
        for k, v in match_result.items():
            print(f"{k}: {v}")
    except FaceMatchError as e:
        print(f"ERROR: {e}")
