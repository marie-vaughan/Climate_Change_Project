import argparse, json, sys
from parser import parse_from_text
from calc import estimate
import easyocr
import cv2
import re
import numpy as np

def resize_if_needed(img, max_dimension=1920):
    """Resize image if too large"""
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

def preprocess_variants(img):
    """Create multiple preprocessing variants"""
    variants = []
    
    # 1. Original
    variants.append((img, 'original'))
    
    # 2. CLAHE Enhanced
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    variants.append((enhanced_bgr, 'clahe'))
    
    # 3. Denoised
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    denoised_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    variants.append((denoised_bgr, 'denoised'))
    
    # 4. Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    variants.append((adaptive_bgr, 'adaptive_thresh'))
    
    # 5. Bilateral Filter
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    bilateral_bgr = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
    variants.append((bilateral_bgr, 'bilateral'))
    
    return variants

def run_easyocr_single(img, reader):
    """Run EasyOCR on a single preprocessed image"""
    try:
        result = reader.readtext(img)
        
        if not result:
            return None, 0
        
        texts = []
        confidences = []
        
        for (bbox, text, conf) in result:
            texts.append(text)
            confidences.append(conf)
        
        full_text = ' '.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return full_text, avg_confidence * 100
        
    except Exception as e:
        return None, 0

def run_easyocr_many(image_paths):
    """Run EasyOCR on multiple images with preprocessing variants"""
    print("Loading EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("✓ EasyOCR loaded")
    
    all_texts = []
    seen = set()
    
    for path in image_paths:
        print(f"\nOCR → {path}")
        
        # Load and resize image
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}")
            continue
        
        img = resize_if_needed(img)
        
        # Try multiple preprocessing variants
        variants = preprocess_variants(img)
        
        best_text = ""
        best_confidence = 0
        
        for preprocessed_img, method_name in variants:
            text, confidence = run_easyocr_single(preprocessed_img, reader)
            
            if text and confidence > best_confidence:
                best_text = text
                best_confidence = confidence
        
        print(f"  Best confidence: {best_confidence:.1f}%")
        
        # Deduplicate and add to merged text
        if best_text:
            words = best_text.split()
            for word in words:
                clean = re.sub(r"\s+", " ", word.strip())
                norm = clean.lower()
                if norm not in seen and len(norm) > 0:
                    seen.add(norm)
                    all_texts.append(clean)
    
    merged_text = " ".join(all_texts)
    return re.sub(r"\s+", " ", merged_text).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--image", type=str, action="append",
                    help="Path to tag image(s); use multiple --image for multi-photo tags")
    ap.add_argument("--weight_g", type=float, default=1000.0)  # Default 1kg
    ap.add_argument("--wash_per_month", type=float, default=2.0)
    ap.add_argument("--mode", type=str, default="ship")
    ap.add_argument("--show_ocr", action="store_true",
                    help="Print merged OCR text before parsing")
    args = ap.parse_args()

    if args.image:
        try:
            # Run EasyOCR on all tag images
            args.text = run_easyocr_many(args.image)

            if args.show_ocr:
                print("\n=== RAW OCR TEXT ===")
                print(args.text)
                print("====================\n")

        except Exception as e:
            print(f"❌ OCR failed: {e}")
            sys.exit(1)

    if not args.text:
        sys.exit("Provide --text or at least one --image")

    record = parse_from_text(
        args.text,
        garment_type=None,  # No longer using garment type
        default_weight_g=args.weight_g,
        washes_per_month=args.wash_per_month
    )
    res = estimate(record, preferred_mode=args.mode)

    print("=== PARSED TAG ===")
    print(json.dumps({
        "materials": [{"fiber": m.fiber, "pct": m.pct} for m in record.materials],
        "origin": record.origin_country,
        "care": vars(record.care)
    }, indent=2))

    print("\n=== RESULTS ===")
    print(json.dumps({
        "total": round(res.total_kgco2e, 3),
        "breakdown": {k: round(v, 3) for k, v in res.breakdown.items()},
        "assumptions": res.assumptions
    }, indent=2))

if __name__ == "__main__":
    main()

