import easyocr
import cv2
import numpy as np
from pathlib import Path

def advanced_preprocess(img):
    """Advanced preprocessing for better OCR"""
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # 4. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh

# Initialize reader
reader = easyocr.Reader(['en'])

# Test on validation set
val_folder = Path('dataset_split/val/images')

for img_path in list(val_folder.glob('*.jpg'))[:5]:
    img = cv2.imread(str(img_path))
    
    # Preprocess
    processed = advanced_preprocess(img)
    
    # Run OCR
    results = reader.readtext(processed)
    
    print(f"\n{img_path.name}:")
    for (bbox, text, prob) in results:
        print(f"  {text} ({prob:.2f})")