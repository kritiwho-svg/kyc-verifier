"""
tamper_detector.py
-------------------
Error Level Analysis (ELA) for spotting digitally forged/edited regions of an ID image.

How it works:
  1. Re-save the (JPEG) image at a known quality level.
  2. Diff the original against the re-saved version, pixel by pixel.
  3. Regions that were NOT re-edited compress predictably and show a low, uniform
     error level. Regions that WERE pasted/edited after the original save were
     compressed a second time at a different quality, so they show an unusually
     high or uneven error level relative to the rest of the image.
  4. We score "tamper likelihood" from the variance/hotspot concentration of the
     ELA map, and return both the heatmap and flagged bounding boxes for the UI.

Note: ELA is a classic, well-known forensic heuristic -- it is not proof of forgery,
only a signal to flag for human review. It works best on JPEGs; PNGs are re-encoded
losslessly so ELA is less informative there (we note this in the result).
"""

import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2


def compute_ela(image_path, quality=90, scale=15):
    """Returns (original_pil, ela_pil, ela_gray_array)."""
    original = Image.open(image_path).convert("RGB")

    tmp_path = image_path + ".__ela_tmp.jpg"
    original.save(tmp_path, "JPEG", quality=quality)
    resaved = Image.open(tmp_path)

    diff = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    max_diff = max(ex[1] for ex in extrema) or 1
    factor = min(scale * 255.0 / max_diff, 40)  # amplify so it's visible to the eye

    ela_image = ImageEnhance.Brightness(diff).enhance(factor)
    ela_gray = np.array(ela_image.convert("L"), dtype=np.float64)

    os.remove(tmp_path)
    return original, ela_image, ela_gray


def _find_hotspots(ela_gray, block=16, z_thresh=2.8, neighborhood=3):
    """Slides a block window over the ELA map and flags blocks whose mean error
    is a statistical outlier relative to their LOCAL neighborhood (not the whole
    image). A pasted/edited patch is inconsistent with the blocks immediately
    around it -- comparing against the whole image instead would get swamped by
    unrelated content (solid color banners, other genuine text, etc.) and miss
    the localized discontinuity that is the actual tamper signature. Flags both
    directions: a splice can show up as unusually high OR unusually low error
    relative to its neighbors depending on its compression history.
    Returns (boxes, global_mean, global_std) for score computation.
    """
    h, w = ela_gray.shape
    grid_h, grid_w = h // block, w // block
    grid = np.zeros((grid_h, grid_w))
    for gy in range(grid_h):
        for gx in range(grid_w):
            y, x = gy * block, gx * block
            grid[gy, gx] = ela_gray[y:y + block, x:x + block].mean()

    boxes = []
    r = neighborhood
    for gy in range(grid_h):
        for gx in range(grid_w):
            y0n, y1n = max(0, gy - r), min(grid_h, gy + r + 1)
            x0n, x1n = max(0, gx - r), min(grid_w, gx + r + 1)
            neighborhood_vals = grid[y0n:y1n, x0n:x1n].flatten()
            # exclude the block itself from its own baseline
            mask = np.ones(neighborhood_vals.shape, dtype=bool)
            self_idx = (gy - y0n) * (x1n - x0n) + (gx - x0n)
            if 0 <= self_idx < len(mask):
                mask[self_idx] = False
            local_vals = neighborhood_vals[mask]
            if len(local_vals) < 4:
                continue
            local_mu, local_sigma = local_vals.mean(), local_vals.std() + 1e-6
            z = (grid[gy, gx] - local_mu) / local_sigma
            if abs(z) > z_thresh:
                boxes.append((gx * block, gy * block, block, block))

    mu, sigma = float(grid.mean()), float(grid.std())
    return _merge_boxes(boxes), mu, sigma


def _merge_boxes(boxes, gap=8):
    """Merges adjacent flagged blocks into larger rectangles for cleaner overlays."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = []
    used = [False] * len(boxes)
    for i, (x, y, w, h) in enumerate(boxes):
        if used[i]:
            continue
        x0, y0, x1, y1 = x, y, x + w, y + h
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, (x2, y2, w2, h2) in enumerate(boxes):
                if used[j]:
                    continue
                bx0, by0, bx1, by1 = x2, y2, x2 + w2, y2 + h2
                if not (bx0 > x1 + gap or bx1 < x0 - gap or by0 > y1 + gap or by1 < y0 - gap):
                    x0, y0 = min(x0, bx0), min(y0, by0)
                    x1, y1 = max(x1, bx1), max(y1, by1)
                    used[j] = True
                    changed = True
        merged.append((x0, y0, x1 - x0, y1 - y0))
    return merged


def analyze_tampering(image_path, quality=90, block=16, z_thresh=2.8):
    """Full ELA pipeline: returns a result dict + saves an annotated overlay image.

    tamper_score (0-100): higher = more likely locally edited/forged.
    """
    ext = os.path.splitext(image_path)[1].lower()
    is_lossy_format = ext in (".jpg", ".jpeg")

    original, ela_image, ela_gray = compute_ela(image_path, quality=quality)
    boxes, mu, sigma = _find_hotspots(ela_gray, block=block, z_thresh=z_thresh)

    # Tamper score is driven purely by how much locally-anomalous area was found.
    # (Global ELA intensity/variance is dominated by unrelated content -- e.g. text
    # density -- and doesn't reliably separate tampered from genuine documents, so
    # it's intentionally not part of this score.)
    total_blocks = max(1, (ela_gray.shape[0] // block) * (ela_gray.shape[1] // block))
    flagged_area_blocks = sum((bw // block) * (bh // block) for (_, _, bw, bh) in boxes)
    flagged_fraction = min(1.0, flagged_area_blocks / max(1, total_blocks * 0.04))
    tamper_score = round(100 * flagged_fraction, 1)

    overlay = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), "..", "outputs")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    ela_path = os.path.join(out_dir, f"{base}_ela.png")
    overlay_path = os.path.join(out_dir, f"{base}_tamper_overlay.png")
    ela_image.save(ela_path)
    cv2.imwrite(overlay_path, overlay)

    verdict = "LIKELY_TAMPERED" if tamper_score >= 20 else (
        "REVIEW_RECOMMENDED" if tamper_score >= 5 else "NO_TAMPERING_DETECTED"
    )

    return {
        "tamper_score": tamper_score,
        "verdict": verdict,
        "flagged_regions": boxes,
        "num_flagged_regions": len(boxes),
        "is_lossy_format": is_lossy_format,
        "ela_image_path": ela_path,
        "overlay_image_path": overlay_path,
        "note": None if is_lossy_format else
                "Input is not JPEG -- ELA is less reliable on lossless formats.",
    }


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/mock_aadhaar_tampered.jpg"
    result = analyze_tampering(path)
    print(json.dumps(result, indent=2, default=str))
