# Document KYC Verifier

A step-by-step KYC verification pipeline: OCR field extraction, face
matching (ID vs selfie), and tamper detection (ELA), combined into a
final verdict (VERIFIED / REJECTED / NEEDS_REVIEW). Streamlit UI included.

⚠️ **Important — read before running:**
This code was written and syntax-checked, and the ELA (forgery) module
was actually executed and validated on test images. However, the
**OCR (EasyOCR) and Face Match (DeepFace/TensorFlow) modules could not
be runtime-tested** in the environment this was built in (no internet
access to install/download the ~1GB of ML packages and model weights).
You must run the checks below yourself on your Windows machine as the
real verification step. If you hit an error, the "Common Errors"
section below covers the ones I anticipated — send me the exact
traceback for anything else.

---

## Folder Structure

```
kyc_verifier/
├── backend/
│   ├── __init__.py
│   ├── ocr.py          # EasyOCR text extraction + Name/DOB/ID parsing
│   ├── face_match.py   # DeepFace (Facenet512) face similarity
│   ├── forgery.py      # ELA-based tamper detection
│   ├── decision.py     # Rule-based verdict engine
│   └── main.py         # verify_kyc() - single pipeline entry point
├── frontend/
│   └── app.py          # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Setup (Windows + venv)

```bat
cd kyc_verifier
python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

This installs TensorFlow + DeepFace + EasyOCR, which is a large
download (~1-1.5GB total) and can take 5-15 minutes depending on your
connection. This is normal.

---

## Step 1: Verify environment

```bat
python -c "import cv2, numpy, PIL, easyocr, streamlit, sklearn; print('base imports OK')"
```

Then test DeepFace separately (this triggers its one-time model download,
~90MB, needs internet):

```bat
python -c "import os; os.environ['TF_USE_LEGACY_KERAS']='1'; from deepface import DeepFace; print('deepface OK')"
```

---

## Step 2: Test each backend module individually

Put a real (mock, no real PII) ID card image and a selfie image
somewhere on disk, then:

```bat
:: OCR
python backend\ocr.py path\to\id_image.jpg

:: Face match
python backend\face_match.py path\to\id_image.jpg path\to\selfie.jpg

:: Forgery / ELA
python backend\forgery.py path\to\id_image.jpg

:: Decision engine (uses built-in sample data, no images needed)
python backend\decision.py
```

---

## Step 3: Test the full pipeline

Run from the **project root** (not inside backend/), using `-m` so the
`backend` package resolves correctly:

```bat
python -m backend.main path\to\id_image.jpg path\to\selfie.jpg
```

This prints a full JSON result: OCR fields, face match score, forgery
score, and final verdict.

---

## Step 4: Run the Streamlit UI

From the project root:

```bat
streamlit run frontend\app.py
```

This opens a browser tab where you can upload an ID + selfie (or use
your webcam) and see the full result: extracted fields, face match
score, ELA visualization, and a verdict badge.

---

## Common Errors & Fixes

**`DLL load failed while importing...` (TensorFlow)**
→ Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), then restart your terminal.

**`ValueError: Only instances of keras.Layer can be added...` (DeepFace)**
→ Means `TF_USE_LEGACY_KERAS=1` wasn't set before DeepFace was imported.
Already handled in `face_match.py`, but if you import deepface anywhere
else first, set that env var before it.

**`ValueError: Face could not be detected...`**
→ Expected behavior — `face_match.py` raises `FaceMatchError` for this,
and `main.py` catches it gracefully (returns similarity_score: 0.0
instead of crashing). Use a clearer, front-facing photo.

**First run is very slow**
→ Normal. EasyOCR and DeepFace both load large models into memory on
first call. Subsequent calls in the same process are much faster.

**Forgery score thresholds feel off**
→ The ELA thresholds in `forgery.py` and `decision.py` are heuristic
starting points. Tamper detection accuracy depends heavily on your
specific ID card image style (photo-heavy vs text-heavy, JPEG quality,
scanner vs. phone camera). Test against a batch of genuine vs.
deliberately-edited samples and adjust the `FORGERY_*` constants in
`decision.py` and the scaling factor in `forgery.py` accordingly.

---

## Notes

- No real PII should be used — test only with mock/sample ID card images.
- `detector_backend="opencv"` is used in `face_match.py` for speed with
  no extra install. If face detection is unreliable on your images, try
  `detector_backend="mtcnn"` or `"retinaface"` (both bundled with DeepFace,
  but slower).
- The decision thresholds in `decision.py` are simple and transparent by
  design — tune the constants at the top of that file as you test.
