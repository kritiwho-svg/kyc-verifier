"""
frontend/app.py
Streamlit UI for the KYC Verifier.

Run from the project root (kyc_verifier/) with:
    streamlit run frontend/app.py
"""

import os
import sys
import tempfile

import streamlit as st

# Make sure the project root is on sys.path so `backend` package imports work
# regardless of the working directory Streamlit was launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import verify_kyc  # noqa: E402


st.set_page_config(page_title="KYC Document Verifier", page_icon="🪪", layout="centered")

st.title("🪪 Document KYC Verifier")
st.caption("Upload a mock ID card and a selfie to run OCR, face match, and tamper detection.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. ID Card")
    id_file = st.file_uploader("Upload ID image", type=["jpg", "jpeg", "png"], key="id_upload")
    if id_file:
        st.image(id_file, caption="ID Card Preview", use_container_width=True)

with col2:
    st.subheader("2. Selfie")
    selfie_source = st.radio("Selfie source", ["Upload", "Webcam"], horizontal=True)
    selfie_file = None
    if selfie_source == "Upload":
        selfie_file = st.file_uploader("Upload selfie", type=["jpg", "jpeg", "png"], key="selfie_upload")
    else:
        selfie_file = st.camera_input("Take a selfie")
    if selfie_file:
        st.image(selfie_file, caption="Selfie Preview", use_container_width=True)

st.divider()

run_button = st.button("🔍 Run KYC Verification", type="primary", use_container_width=True)

if run_button:
    if not id_file or not selfie_file:
        st.error("Please provide both an ID image and a selfie before running verification.")
    else:
        with st.spinner("Running OCR, face match, and forgery detection... this may take a moment."):
            # Save uploaded files to a temp dir so backend modules (which expect
            # file paths) can read them.
            tmp_dir = tempfile.mkdtemp()
            id_path = os.path.join(tmp_dir, "id_image.jpg")
            selfie_path = os.path.join(tmp_dir, "selfie_image.jpg")
            ela_path = os.path.join(tmp_dir, "ela_output.jpg")

            with open(id_path, "wb") as f:
                f.write(id_file.getbuffer())
            with open(selfie_path, "wb") as f:
                f.write(selfie_file.getbuffer())

            try:
                result = verify_kyc(id_path, selfie_path, ela_output_path=ela_path)
            except Exception as e:
                st.error(f"Verification pipeline crashed: {e}")
                st.stop()

        st.success("Verification complete.")

        # --- Verdict badge ---
        verdict = result["decision"]["verdict"]
        reasons = result["decision"]["reasons"]

        badge_colors = {
            "VERIFIED": "✅",
            "REJECTED": "❌",
            "NEEDS_REVIEW": "⚠️",
        }
        st.header(f"{badge_colors.get(verdict, '')} {verdict}")
        for r in reasons:
            st.write(f"- {r}")

        if result["errors"]:
            st.warning("Pipeline warnings:")
            for e in result["errors"]:
                st.write(f"- {e}")

        st.divider()

        # --- Details in columns ---
        detail_col1, detail_col2, detail_col3 = st.columns(3)

        with detail_col1:
            st.subheader("📄 Extracted Fields")
            ocr = result["ocr"]
            st.write(f"**Name:** {ocr['name']}")
            st.write(f"**DOB:** {ocr['dob']}")
            st.write(f"**ID Number:** {ocr['id_number']}")
            st.write(f"**OCR Confidence:** {ocr['avg_confidence']}")
            with st.expander("Raw OCR text"):
                st.text(ocr["raw_text"])

        with detail_col2:
            st.subheader("🙂 Face Match")
            face = result["face_match"]
            st.metric("Similarity Score", f"{face['similarity_score']} / 100")
            st.write(f"**DeepFace verified:** {face['verified']}")
            st.write(f"**Distance:** {face['distance']}")

        with detail_col3:
            st.subheader("🕵️ Forgery (ELA)")
            forgery = result["forgery"]
            st.metric("Forgery Score", f"{forgery['forgery_score']} / 100")
            st.write(f"**Mean ELA intensity:** {forgery['mean_ela_intensity']}")

        if result["forgery"].get("ela_image_path") and os.path.exists(result["forgery"]["ela_image_path"]):
            st.subheader("ELA Visualization")
            st.image(result["forgery"]["ela_image_path"], caption="Error Level Analysis output", use_container_width=True)
