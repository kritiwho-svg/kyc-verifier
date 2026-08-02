"""
app.py
------
Flask demo UI for the KYC verifier pipeline. Upload a mock ID + a selfie,
get back OCR fields, a face-match score, an ELA tamper heatmap, and an
overall verdict -- rendered visually for a hackathon judging demo.

Run:
    python3 app.py
Then open http://localhost:5000
"""

import os
import uuid
from flask import Flask, request, render_template, redirect, url_for, send_from_directory

from pipeline import run_kyc_verification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "outputs", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB


@app.route("/")
def index():
    samples = {
        "id_genuine": "sample_data/mock_aadhaar_genuine.jpg",
        "id_tampered": "sample_data/mock_aadhaar_tampered.jpg",
        "selfie_match": "sample_data/selfie_match.jpg",
        "selfie_mismatch": "sample_data/selfie_mismatch.jpg",
    }
    return render_template("index.html", samples=samples)


@app.route("/verify", methods=["POST"])
def verify():
    id_file = request.files.get("id_image")
    selfie_file = request.files.get("selfie_image")

    id_sample = request.form.get("id_sample")
    selfie_sample = request.form.get("selfie_sample")

    session_id = uuid.uuid4().hex[:8]

    if id_file and id_file.filename:
        id_path = os.path.join(UPLOAD_DIR, f"{session_id}_id_{id_file.filename}")
        id_file.save(id_path)
    else:
        id_path = os.path.join(BASE_DIR, id_sample)

    if selfie_file and selfie_file.filename:
        selfie_path = os.path.join(UPLOAD_DIR, f"{session_id}_selfie_{selfie_file.filename}")
        selfie_file.save(selfie_path)
    else:
        selfie_path = os.path.join(BASE_DIR, selfie_sample)

    result = run_kyc_verification(id_path, selfie_path)

    # make paths servable via /files/<name>
    result["_id_image_rel"] = _to_served_path(id_path)
    result["_selfie_image_rel"] = _to_served_path(selfie_path)
    result["_ela_image_rel"] = _to_served_path(result["tamper_check"]["ela_image_path"])
    result["_overlay_image_rel"] = _to_served_path(result["tamper_check"]["overlay_image_path"])

    return render_template("result.html", r=result)


def _to_served_path(abs_path):
    """Maps an absolute file path under the project dir to a /files/... URL."""
    rel = os.path.relpath(abs_path, BASE_DIR)
    return url_for("files", filename=rel.replace(os.sep, "/"))


@app.route("/files/<path:filename>")
def files(filename):
    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
