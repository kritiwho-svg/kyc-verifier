"""
face_match.py
--------------
Face detection + embedding-similarity cross-check between an ID photo and a live selfie.

Design:
  - If the `face_recognition` (dlib-based FaceNet-style) library is installed, it is used
    for state-of-the-art 128-d embeddings -- install it for the "real" hackathon demo:
        pip install face_recognition dlib
  - Otherwise, falls back to a fully offline OpenCV-only pipeline:
        Haar cascade face detection -> aligned crop -> HOG descriptor embedding
        -> cosine similarity
    This keeps the demo runnable with zero extra downloads/model weights.

Either backend exposes the same `compare_faces(id_image_path, selfie_path)` API.
"""

import cv2
import numpy as np

try:
    import face_recognition  # noqa: F401
    _BACKEND = "face_recognition"
except Exception:
    # Covers both "not installed" (ImportError) and "installed but broken"
    # (e.g. dlib compiled without a working shared library) -- either way,
    # fall back to the offline backend rather than crashing the app.
    _BACKEND = "opencv_hog"

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

# Similarity thresholds are backend-specific since the two methods produce
# scores on different scales.
THRESHOLDS = {
    "face_recognition": 0.60,   # 1 - dlib face_distance; >= this => same person
    "opencv_hog": 0.55,         # cosine similarity of HOG descriptors
}


def _detect_face_bbox(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(60, 60))
    if len(faces) == 0:
        return None
    # pick the largest detected face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]  # x, y, w, h


def _crop_face(image_bgr, bbox, size=128, margin=0.25):
    x, y, w, h = bbox
    mx, my = int(w * margin), int(h * margin)
    H, W = image_bgr.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    crop = image_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size))


def _hog_embedding(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    hog = cv2.HOGDescriptor(_winSize=(128, 128), _blockSize=(32, 32),
                             _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9)
    vec = hog.compute(gray)
    return vec.flatten()


def _cosine_sim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _center_square_bbox(img):
    """Last-resort fallback when Haar cascade finds no face (e.g. a tightly
    pre-cropped photo, illustration, or low-quality scan): assume the subject
    roughly fills the frame and use a centered square region instead of failing
    outright. Callers should treat this case as lower-confidence."""
    h, w = img.shape[:2]
    side = int(min(h, w) * 0.9)
    x = (w - side) // 2
    y = (h - side) // 2
    return (x, y, side, side)


def get_face_embedding(image_path, allow_fallback_crop=True):
    """Returns (embedding_vector, bbox, backend_name, detection_method) for the
    largest face found. detection_method is 'haar'/'dlib' for a real detection,
    or 'center_crop_fallback' if no face detector fired and we fell back to a
    centered crop. Returns (None, None, backend_name, None) if nothing usable.

    NOTE ON bbox FORMAT (differs by backend -- don't assume one shape downstream):
      - face_recognition backend: (top, right, bottom, left)
      - opencv_hog backend:       (x, y, w, h)
    """
    if _BACKEND == "face_recognition":
        import face_recognition as fr
        img = fr.load_image_file(image_path)
        # ID-card photos are often small -- upsample once to help the detector
        # find faces it would otherwise miss on a low-res thumbnail crop.
        locations = fr.face_locations(img, number_of_times_to_upsample=1, model="hog")
        if locations:
            # face_locations returns (top, right, bottom, left) tuples; pick the
            # largest by area in case more than one face is in frame (e.g. a
            # selfie with a bystander in the background).
            def _area(loc):
                top, right, bottom, left = loc
                return max(0, bottom - top) * max(0, right - left)
            best_loc = max(locations, key=_area)
            encodings = fr.face_encodings(img, known_face_locations=[best_loc])
            return encodings[0], best_loc, _BACKEND, "dlib"
        if not allow_fallback_crop:
            return None, None, _BACKEND, None
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None, None, _BACKEND, None
        bbox = _center_square_bbox(img_bgr)
        crop = _crop_face(img_bgr, bbox)
        embedding = _hog_embedding(crop)
        return embedding, bbox, _BACKEND, "center_crop_fallback"

    # OpenCV fallback backend
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    bbox = _detect_face_bbox(img)
    method = "haar"
    if bbox is None:
        if not allow_fallback_crop:
            return None, None, _BACKEND, None
        bbox = _center_square_bbox(img)
        method = "center_crop_fallback"
    face_crop = _crop_face(img, bbox)
    embedding = _hog_embedding(face_crop)
    return embedding, bbox, _BACKEND, method


def compare_faces(id_image_path, selfie_path):
    """Cross-checks the face on the ID document against the live selfie.

    Returns a dict with: match (bool), similarity (float 0-1), backend, and any
    detection errors so the caller can render a clear judge-facing verdict.
    """
    result = {
        "backend": _BACKEND,
        "id_face_found": False,
        "selfie_face_found": False,
        "similarity": None,
        "match": False,
        "verdict": "ERROR",
        "detail": "",
    }

    id_emb, id_bbox, backend, id_method = get_face_embedding(id_image_path)
    selfie_emb, selfie_bbox, _, selfie_method = get_face_embedding(selfie_path)

    result["id_face_found"] = id_emb is not None
    result["selfie_face_found"] = selfie_emb is not None
    result["id_bbox"] = id_bbox
    result["selfie_bbox"] = selfie_bbox
    result["id_detection_method"] = id_method
    result["selfie_detection_method"] = selfie_method
    if id_method == "center_crop_fallback" or selfie_method == "center_crop_fallback":
        result["low_confidence_detection"] = True

    if id_emb is None or selfie_emb is None:
        result["verdict"] = "NO_FACE_DETECTED"
        result["detail"] = "Could not detect a face in one or both images."
        return result

    if backend == "face_recognition":
        import face_recognition as fr
        distance = fr.face_distance([id_emb], selfie_emb)[0]
        similarity = 1.0 - float(distance)
    else:
        similarity = _cosine_sim(id_emb, selfie_emb)

    threshold = THRESHOLDS[backend]
    is_match = similarity >= threshold

    result["similarity"] = round(similarity, 4)
    result["threshold"] = threshold
    result["match"] = bool(is_match)
    result["verdict"] = "MATCH" if is_match else "MISMATCH"
    result["detail"] = (
        f"Similarity {similarity:.3f} vs threshold {threshold} using {backend} backend."
    )
    return result


if __name__ == "__main__":
    import sys, json
    id_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/mock_aadhaar_genuine.jpg"
    selfie_path = sys.argv[2] if len(sys.argv) > 2 else "sample_data/selfie_match.jpg"
    print(json.dumps(compare_faces(id_path, selfie_path), indent=2, default=str))
