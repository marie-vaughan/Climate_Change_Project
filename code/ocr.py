"""
OCR module for garment tags using PaddleOCR ≥ 2.7.
Works far better than Tesseract on multi-language, low-contrast, or curved tags.
"""

import os, re
import numpy as np
import cv2
from paddleocr import PaddleOCR

ocr_model = PaddleOCR(lang="en", use_textline_orientation=True)

def _prepare(path, max_side=1800):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    # Downscale only if really large
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Convert to grayscale (reduces fabric color noise)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gentle denoise to smooth fabric weave
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Mild contrast stretch instead of CLAHE (keeps flat background)
    min_val, max_val = gray.min(), gray.max()
    if max_val > min_val:
        gray = ((gray - min_val) / (max_val - min_val) * 255).astype("uint8")

    # Light sharpening (helps text edges)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    # Back to 3-channel for PaddleOCR
    prep = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return prep

def run_ocr_many(image_paths, dedupe=True):
    image_paths = [p for p in image_paths if os.path.exists(p)]
    if not image_paths:
        raise FileNotFoundError("No valid image paths provided.")

    merged, seen = [], set()
    for path in image_paths:
        img = _prepare(path, max_side=1600)
        print(f"OCR → {path}  (resized/enhanced)")
        results = ocr_model.ocr(img)  # ndarray input; no 'cls='
        for page in results:
            for line in page:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                txt, conf = line[1]
                if conf and conf > 0.25:
                    clean = re.sub(r"\s+", " ", txt.strip())
                    if dedupe:
                        norm = clean.lower()
                        if norm in seen:
                            continue
                        seen.add(norm)
                    merged.append(clean)

    text = " ".join(merged)
    return re.sub(r"\s+", " ", text).strip()