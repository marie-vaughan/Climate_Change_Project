# ocr_viz.py — safe visualizer that resizes before OCR to avoid OOM/kills.
import sys, os, re
import numpy as np
import cv2
from paddleocr import PaddleOCR

ocr_model = PaddleOCR(lang="en", use_textline_orientation=True, det_db_thresh=0.1, det_db_box_thresh=0.2)

def load_and_prepare(path, max_side=1800):
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python code/ocr_viz.py path/to/image.jpg")
        sys.exit(2)

    in_path = sys.argv[1]
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        sys.exit(2)

    # CRITICAL: prepare image BEFORE OCR to avoid OOM
    img = load_and_prepare(in_path, max_side=1600)

    # PaddleOCR accepts numpy arrays directly; no need to write a temp file
    results = ocr_model.ocr(img)  # do NOT pass cls=..., new API handles orientation internally

    # Overlay boxes + confidences
    vis = img.copy()
    hits = 0
    for page in results:
        for line in page:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            box, (txt, conf) = line
            if conf is None or conf < 0.25:
                continue
            pts = np.array(box, dtype=int)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cx, cy = pts[0]
            cv2.putText(vis, f"{conf:.2f}", (int(cx), int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            hits += 1

    out_path = re.sub(r'(\.\w+)$', r'_ocr\1', in_path)
    cv2.imwrite(out_path, vis)
    print(f"Saved visualization with {hits} boxes → {out_path}")

    # Print extracted text so you can inspect it
    chunks = []
    for page in results:
        for line in page:
            # Each element can be [box, (txt, conf)]  OR  [box, [txt, conf]]
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            box = line[0]
            txt, conf = line[1][0], line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else (None, None)
            if conf and conf >= 0.25:
                chunks.append(re.sub(r"\s+", " ", str(txt).strip()))
    print(" ".join(chunks))
    print("-------------------------------------")

if __name__ == "__main__":
    main()

