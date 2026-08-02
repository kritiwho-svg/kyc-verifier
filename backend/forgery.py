"""
forgery.py
Error Level Analysis (ELA) based tampering detection.

How it works:
1. Re-save the image at a known JPEG quality (e.g. 90).
2. Compute the pixel-wise difference between original and re-saved version.
3. Amplify the difference -- areas that were edited/pasted compress
   differently than the rest of the (untouched) image, so they show up
   as brighter regions in the ELA output.
4. We use the mean brightness of the ELA image as a rough tampering score.

This is a heuristic signal, not a legal-grade forensic tool.
"""

import os
from PIL import Image, ImageChops
import numpy as np


def generate_ela_image(image_path: str, output_path: str, quality: int = 90) -> str:
    """
    Generates and saves an ELA image. Returns the output path.
    """
    original = Image.open(image_path).convert("RGB")

    temp_path = output_path + ".temp.jpg"
    original.save(temp_path, "JPEG", quality=quality)
    resaved = Image.open(temp_path)

    ela_image = ImageChops.difference(original, resaved)

    extrema = ela_image.getextrema()
    max_diff = max(ex[1] for ex in extrema)
    if max_diff == 0:
        max_diff = 1  # avoid div-by-zero for a perfectly flat image

    scale = 255.0 / max_diff
    ela_image = ela_image.point(lambda p: p * scale)

    ela_image.save(output_path, "JPEG")

    os.remove(temp_path)
    return output_path


def get_forgery_score(image_path: str, output_path: str = None, quality: int = 90) -> dict:
    """
    Returns:
        {
            "forgery_score": float (0-100, HIGHER = more likely tampered),
            "ela_image_path": str,
            "mean_ela_intensity": float,
            "max_ela_intensity": float
        }
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_ela.jpg"

    generate_ela_image(image_path, output_path, quality=quality)

    ela_array = np.array(Image.open(output_path).convert("L")).astype(float)  # grayscale
    mean_intensity = float(np.mean(ela_array))
    max_intensity = float(np.max(ela_array))
    # 95th percentile is more sensitive to small localized edits (e.g. a
    # pasted-in text field) than the plain mean, which gets diluted by
    # large untouched regions of the image.
    p95_intensity = float(np.percentile(ela_array, 95))

    # Heuristic scaling based on p95 intensity. Genuine, untouched JPEGs
    # typically show low p95 values; localized tampering pushes this up.
    # NOTE: this is a heuristic starting point, not a calibrated forensic
    # threshold. ELA is also naturally noisier on images with lots of
    # sharp text/graphic edges (common on ID cards) -- always tune these
    # bands against a real batch of genuine vs. known-tampered samples
    # before relying on this in production.
    forgery_score = min(100.0, (p95_intensity / 35.0) * 100)

    return {
        "forgery_score": round(forgery_score, 2),
        "ela_image_path": output_path,
        "mean_ela_intensity": round(mean_intensity, 3),
        "p95_ela_intensity": round(p95_intensity, 3),
        "max_ela_intensity": round(max_intensity, 3),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python forgery.py <path_to_image>")
        sys.exit(1)

    path = sys.argv[1]
    result = get_forgery_score(path)
    print("\n--- FORGERY (ELA) RESULT ---")
    for k, v in result.items():
        print(f"{k}: {v}")
