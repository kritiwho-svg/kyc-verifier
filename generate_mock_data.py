"""
generate_mock_data.py
----------------------
Generates 100% synthetic mock KYC assets for demoing the pipeline:
  - A mock Aadhaar-style ID card image
  - A mock PAN-style ID card image
  - A "genuine" tampered version of the Aadhaar card (edited name field) for ELA testing
  - Two synthetic face images (a "matching" pair and a "mismatching" pair) so the
    face-match module has something to run against without needing real photographs.

No real person's data is used anywhere -- all names/numbers/faces are fabricated.
Run this once before the demo:  python3 generate_mock_data.py
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(OUT_DIR, exist_ok=True)

FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size, bold=False):
    return ImageFont.truetype(FONT_PATH_BOLD if bold else FONT_PATH_REG, size)


def _draw_synthetic_face(draw, cx, cy, r, skin=(222, 184, 155), seed=0):
    """Draws a simple cartoon face used as a stand-in for a real photograph."""
    rnd = random.Random(seed)
    jitter = lambda: rnd.randint(-4, 4)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=skin, outline=(90, 60, 40))
    eye_y = cy - r // 4
    for dx in (-r // 2, r // 2):
        ex, ey = cx + dx + jitter(), eye_y + jitter()
        draw.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(30, 30, 30))
    draw.arc([cx - r // 2, cy + r // 6, cx + r // 2, cy + r // 1.5], start=20, end=160,
              fill=(120, 60, 60), width=4)
    draw.ellipse([cx - r - 10, cy - 10, cx - r + 15, cy + 20], fill=skin, outline=(90, 60, 40))
    draw.ellipse([cx + r - 15, cy - 10, cx + r + 10, cy + 20], fill=skin, outline=(90, 60, 40))


def make_face(path, seed, skin=(222, 184, 155)):
    img = Image.new("RGB", (300, 300), (235, 235, 240))
    d = ImageDraw.Draw(img)
    _draw_synthetic_face(d, 150, 160, 90, skin=skin, seed=seed)
    img.save(path, quality=95)
    return path


def make_aadhaar_card(path, name, dob, aadhaar_no, gender, face_seed):
    img = Image.new("RGB", (856, 540), (240, 248, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 855, 70], fill=(20, 90, 160))
    d.text((20, 15), "GOVERNMENT OF INDIA (MOCK / SYNTHETIC - NOT A REAL DOCUMENT)",
            font=_font(18, bold=True), fill=(255, 255, 255))
    _draw_synthetic_face(d, 130, 260, 80, seed=face_seed)
    d.text((250, 140), f"Name: {name}", font=_font(24, bold=True), fill=(20, 20, 20))
    d.text((250, 190), f"DOB: {dob}", font=_font(22), fill=(20, 20, 20))
    d.text((250, 230), f"Gender: {gender}", font=_font(22), fill=(20, 20, 20))
    d.text((250, 290), f"Aadhaar No: {aadhaar_no}", font=_font(26, bold=True), fill=(10, 10, 90))
    d.rectangle([0, 480, 855, 540], fill=(20, 90, 160))
    d.text((20, 495), "SYNTHETIC MOCK DATA -- FOR HACKATHON DEMO USE ONLY",
            font=_font(16), fill=(255, 255, 255))
    img.save(path, quality=92)
    return path


def make_pan_card(path, name, father_name, dob, pan_no, face_seed):
    img = Image.new("RGB", (856, 540), (255, 250, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 855, 70], fill=(150, 40, 40))
    d.text((20, 15), "INCOME TAX DEPT (MOCK / SYNTHETIC - NOT A REAL DOCUMENT)",
            font=_font(18, bold=True), fill=(255, 255, 255))
    _draw_synthetic_face(d, 130, 260, 80, seed=face_seed)
    d.text((250, 120), f"Permanent Account Number", font=_font(16), fill=(80, 80, 80))
    d.text((250, 145), f"{pan_no}", font=_font(28, bold=True), fill=(10, 10, 90))
    d.text((250, 200), f"Name: {name}", font=_font(22, bold=True), fill=(20, 20, 20))
    d.text((250, 235), f"Father's Name: {father_name}", font=_font(20), fill=(20, 20, 20))
    d.text((250, 270), f"DOB: {dob}", font=_font(20), fill=(20, 20, 20))
    img.save(path, quality=92)
    return path


def simulate_scan(path, blur_radius=0.6, quality=85):
    """Makes a cleanly-rendered vector image look like a real scanned/photographed
    document: slight blur + a single JPEG compression pass. This gives the image a
    realistic, UNIFORM baseline noise floor -- without this step, freshly-rendered
    vector text has uniformly sharp edges everywhere, which makes it impossible to
    tell edited text apart from original text using ELA. Real documents don't have
    this problem because they only go through one physical scan/photograph."""
    img = Image.open(path).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    img.save(path, "JPEG", quality=quality)
    return path


def make_tampered_copy(src_path, dst_path, box, new_text, font_size=24, splice_source=None):
    """Simulates a forged document via the classic 'splice' scenario ELA is designed
    to catch: a patch is rendered independently (its own encode/decode history),
    then pasted into the target image and the WHOLE thing is re-saved once. The
    spliced region has a different underlying compression fingerprint than the area
    around it, which is exactly the discontinuity ELA looks for -- unlike a same-
    style text overwrite on an already-uniform background, which barely differs
    from the surrounding text's own edge noise."""
    img = Image.open(src_path).convert("RGB")

    # Render the replacement text on its own small canvas, independently JPEG-
    # compressed at a different quality, then paste it in. That independent
    # compression pass is what creates the detectable ELA discontinuity.
    x0, y0, x1, y1 = box
    patch = Image.new("RGB", (x1 - x0, y1 - y0), (240, 248, 255))
    pd = ImageDraw.Draw(patch)
    pd.text((5, 5), new_text, font=_font(font_size, bold=True), fill=(20, 20, 20))
    patch_tmp = dst_path + ".patch.jpg"
    patch.save(patch_tmp, "JPEG", quality=97)  # crisp, high-quality patch pasted onto a softer base
    patch = Image.open(patch_tmp).convert("RGB")
    os.remove(patch_tmp)

    img.paste(patch, (x0, y0))
    img.save(dst_path, "JPEG", quality=92)
    return dst_path


if __name__ == "__main__":
    aadhaar_path = os.path.join(OUT_DIR, "mock_aadhaar_genuine.jpg")
    pan_path = os.path.join(OUT_DIR, "mock_pan_genuine.jpg")
    tampered_path = os.path.join(OUT_DIR, "mock_aadhaar_tampered.jpg")
    selfie_match_path = os.path.join(OUT_DIR, "selfie_match.jpg")
    selfie_mismatch_path = os.path.join(OUT_DIR, "selfie_mismatch.jpg")

    make_aadhaar_card(aadhaar_path, "RAVI KUMAR SHARMA", "15-08-1995",
                       "2345 6789 0123", "Male", face_seed=42)
    make_pan_card(pan_path, "RAVI KUMAR SHARMA", "SURESH SHARMA", "15-08-1995",
                  "ABCDE1234F", face_seed=42)
    simulate_scan(aadhaar_path)
    simulate_scan(pan_path)

    # Tampered doc: forger overwrote the name field (crisp new text pasted onto the
    # already-scanned/blurred base) -> creates a local ELA anomaly at that region
    make_tampered_copy(aadhaar_path, tampered_path, box=[248, 138, 620, 172],
                        new_text="AMIT KUMAR VERMA")

    # Selfie that matches the ID photo's synthetic seed -> same face params
    make_face(selfie_match_path, seed=42)
    # Selfie from a different identity -> should be flagged as mismatch
    make_face(selfie_mismatch_path, seed=99, skin=(180, 140, 110))

    print("Mock data generated in:", OUT_DIR)
    for f in os.listdir(OUT_DIR):
        print(" -", f)
