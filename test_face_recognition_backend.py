"""
test_face_recognition_backend.py
----------------------------------
Two things live here:

1. `test_integration_logic_with_mock()` -- runs with NO real dependencies. It
   installs a fake `face_recognition` module into sys.modules that mimics the
   real API's shape (face_locations / face_encodings / face_distance) and
   drives face_match.py's code path through it. This catches integration bugs
   (wrong argument order, wrong tuple unpacking, wrong distance-vs-similarity
   math) without needing dlib installed -- useful in CI or an offline sandbox.
   It does NOT tell you anything about real-world face-match accuracy.

2. `test_with_real_library()` -- run this one after `pip install face_recognition
   dlib` to validate against the actual library on real photos. This is the
   one that matters for your hackathon demo; run it locally where you have
   internet access to install dlib.
"""

import sys
import types
import importlib
import numpy as np


def test_integration_logic_with_mock():
    print("=== Mock face_recognition integration test (no dlib required) ===")

    fake_fr = types.ModuleType("face_recognition")

    # Two fixed 128-d "embeddings" so we can control exactly what similarity
    # comes out the other end and assert on it.
    EMB_A = np.ones(128) * 0.1
    EMB_A_CLOSE = EMB_A + 0.001   # should count as a MATCH
    EMB_B_FAR = np.ones(128) * 5.0  # should count as a MISMATCH

    state = {"call": 0}

    def load_image_file(path):
        return np.zeros((200, 200, 3), dtype=np.uint8)

    def face_locations(img, number_of_times_to_upsample=1, model="hog"):
        # Return two candidate faces to exercise the "pick the largest" logic.
        return [(10, 60, 60, 10), (0, 15, 15, 0)]  # (top, right, bottom, left)

    def face_encodings(img, known_face_locations=None):
        state["call"] += 1
        # First image processed -> EMB_A ; second -> EMB_A_CLOSE (match case)
        return [EMB_A if state["call"] == 1 else EMB_A_CLOSE]

    def face_distance(known, candidate):
        return [float(np.linalg.norm(np.array(known[0]) - np.array(candidate)))]

    fake_fr.load_image_file = load_image_file
    fake_fr.face_locations = face_locations
    fake_fr.face_encodings = face_encodings
    fake_fr.face_distance = face_distance

    sys.modules["face_recognition"] = fake_fr

    # Force face_match to re-detect the (now mocked) library.
    if "face_match" in sys.modules:
        del sys.modules["face_match"]
    import face_match
    importlib.reload(face_match)

    assert face_match._BACKEND == "face_recognition", (
        f"Expected mock to be picked up as backend, got {face_match._BACKEND}"
    )

    # --- Case 1: matching pair ---
    result = face_match.compare_faces("fake_id.jpg", "fake_selfie.jpg")
    assert result["backend"] == "face_recognition"
    assert result["id_face_found"] and result["selfie_face_found"]
    assert result["id_detection_method"] == "dlib"
    assert result["verdict"] == "MATCH", f"Expected MATCH, got {result}"
    print("PASS: matching-embedding case resolves to MATCH ->", result["similarity"])

    # --- Case 2: mismatching pair (swap in the far embedding) ---
    state["call"] = 0
    def face_encodings_mismatch(img, known_face_locations=None):
        state["call"] += 1
        return [EMB_A if state["call"] == 1 else EMB_B_FAR]
    fake_fr.face_encodings = face_encodings_mismatch

    result2 = face_match.compare_faces("fake_id.jpg", "fake_selfie2.jpg")
    assert result2["verdict"] == "MISMATCH", f"Expected MISMATCH, got {result2}"
    print("PASS: far-embedding case resolves to MISMATCH ->", result2["similarity"])

    # --- Case 3: no face detected -> should hit the center-crop fallback,
    #     not crash ---
    fake_fr.face_locations = lambda img, **kw: []
    # give it a real file to fall back to (fallback path calls cv2.imread)
    result3 = face_match.compare_faces(
        "sample_data/mock_aadhaar_genuine.jpg", "sample_data/selfie_match.jpg"
    )
    assert result3["id_detection_method"] == "center_crop_fallback"
    print("PASS: no-detection case falls back to center-crop without crashing ->",
          result3["verdict"])

    del sys.modules["face_recognition"]
    if "face_match" in sys.modules:
        del sys.modules["face_match"]
    print("All mock integration checks passed.\n")


def test_with_real_library():
    """Run this after `pip install face_recognition dlib` on real photos."""
    print("=== Real face_recognition/dlib test ===")
    try:
        import face_recognition  # noqa: F401
    except ImportError:
        print("SKIPPED: face_recognition/dlib is not installed.")
        print("Run: pip install face_recognition dlib")
        return

    if "face_match" in sys.modules:
        del sys.modules["face_match"]
    import face_match
    importlib.reload(face_match)

    assert face_match._BACKEND == "face_recognition", (
        "face_recognition is importable but face_match didn't select it -- "
        "check for an exception during import (see face_match.py's try/except)."
    )
    print("Backend confirmed:", face_match._BACKEND)

    import os
    id_path = "sample_data/mock_aadhaar_genuine.jpg"
    selfie_path = "sample_data/selfie_match.jpg"
    if not (os.path.exists(id_path) and os.path.exists(selfie_path)):
        print("SKIPPED: run generate_mock_data.py first, or point this at real photos.")
        return

    result = face_match.compare_faces(id_path, selfie_path)
    print("Result on bundled mock data (expect NO_FACE_DETECTED -- these are")
    print("cartoon placeholders, not real faces; that's expected, not a failure):")
    import json
    print(json.dumps(result, indent=2, default=str))
    print()
    print("For a meaningful accuracy check, edit this function to point at two")
    print("real photos of the same person, and two of different people.")


if __name__ == "__main__":
    test_integration_logic_with_mock()
    test_with_real_library()
