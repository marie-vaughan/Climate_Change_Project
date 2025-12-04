# evaluate_ocr_easyocr.py
import easyocr
import cv2
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
import numpy as np
from codecarbon import EmissionsTracker
import os

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

def load_ground_truth(txt_path):
    """Load ground truth from txt file"""
    ground_truth = {'country': None, 'material': None, 'care': None}
    
    if not Path(txt_path).exists():
        return ground_truth
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('COUNTRY:'):
                value = line.replace('COUNTRY:', '').strip()
                ground_truth['country'] = None if value == "not visible" else value
            elif line.startswith('MATERIAL:'):
                value = line.replace('MATERIAL:', '').strip()
                ground_truth['material'] = None if value == "not visible" else value
            elif line.startswith('CARE:'):
                value = line.replace('CARE:', '').strip()
                ground_truth['care'] = None if value == "not visible" else value
    
    return ground_truth

def extract_country(text):
    """Extract country from OCR text"""
    if not text or text.strip() == '':
        return "not visible"
    
    patterns = [
        r'MADE IN\s+([A-Z\s]+?)(?:\s+FABRIQUE|\s+HECHO|\s+MADE FOR|RN|$)',
        r'Made In\s+([A-Za-z\s]+?)(?:\s+Fabrique|\s+Hecho|\s+Made for|RN|$)',
        r'FABRIQUÉ EN\s+([A-Z\s]+?)(?:\s+MADE|$)',
        r'Hecho en\s+([A-Za-z\s]+?)(?:\s+Made|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            country = match.group(1).strip()
            country = re.split(r'\s+(?:LA|EN)', country)[0].strip()
            return country
    
    return "not visible"

def extract_material(text):
    """Extract material composition from OCR text"""
    if not text or text.strip() == '':
        return "not visible"
    
    pattern = r'\d+%\s*[A-Za-z]+(?:\s+[A-Za-z]+)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if matches:
        return ' '.join(matches)
    
    return "not visible"

def extract_care(text):
    """Extract care instructions from OCR text"""
    if not text or text.strip() == '':
        return "not visible"
    
    care_pattern = r'(?:HAND WASH|MACHINE WASH|DRY CLEAN|DO NOT|WASH|TUMBLE)(.*?)(?=MADE IN|MADE FOR|FABRIQUÉ|RN#|\d+%|$)'
    match = re.search(care_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        care_text = match.group(0).strip()
        lines = care_text.split('\n')
        english_lines = []
        
        french_indicators = ['À', 'É', 'LAVER', 'FROIDE', 'PAS', 'SÉCHER', 'REPASSAGE']
        
        for line in lines:
            if not any(indicator in line.upper() for indicator in french_indicators):
                if line.strip():
                    english_lines.append(line.strip())
        
        result = ' '.join(english_lines) if english_lines else None
        return result if result else "not visible"
    
    return "not visible"

def run_easyocr_single(img, reader):
    """Run EasyOCR on a single preprocessed image"""
    try:
        # EasyOCR returns: [[bbox, text, confidence], ...]
        result = reader.readtext(img)
        
        if not result:
            return None, 0
        
        # Extract text and confidence
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

def normalize_text(text):
    """Normalize text for comparison"""
    if text is None or text == "not visible":
        return None
    text = str(text).lower().strip()
    text = text.rstrip('.,;')
    text = re.sub(r'\s+', ' ', text)
    return text

def fuzzy_match_score(str1, str2):
    """Calculate similarity between two strings (0-1)"""
    if str1 is None or str2 is None:
        return 0.0
    if str1 == "not visible" or str2 == "not visible":
        if str1 == str2:
            return 1.0
        return 0.0
    
    str1_norm = normalize_text(str1)
    str2_norm = normalize_text(str2)
    
    if str1_norm is None or str2_norm is None:
        return 0.0
    
    return SequenceMatcher(None, str1_norm, str2_norm).ratio()

def calculate_combined_fuzzy_score(prediction, ground_truth):
    """Calculate combined fuzzy score for country, material, and care"""
    country_score = fuzzy_match_score(prediction['country'], ground_truth['country'])
    material_score = fuzzy_match_score(prediction['material'], ground_truth['material'])
    care_score = fuzzy_match_score(prediction['care'], ground_truth['care'])
    
    combined_score = (country_score + material_score + care_score) / 3
    
    return combined_score, {
        'country': country_score,
        'material': material_score,
        'care': care_score
    }

def run_easyocr_ensemble(img, reader, ground_truth):
    """Run EasyOCR on multiple preprocessing variants"""
    variants = preprocess_variants(img)
    
    results = []
    
    for preprocessed_img, method_name in variants:
        text, confidence = run_easyocr_single(preprocessed_img, reader)
        
        if text is None:
            text = ''
        
        # Extract structured information
        country = extract_country(text)
        material = extract_material(text)
        care = extract_care(text)
        
        prediction = {
            'country': country,
            'material': material,
            'care': care
        }
        
        # Calculate fuzzy score against ground truth
        combined_score, individual_scores = calculate_combined_fuzzy_score(
            prediction, ground_truth
        )
        
        results.append({
            'text': text,
            'confidence': confidence,
            'method': method_name,
            'country': country,
            'material': material,
            'care': care,
            'combined_fuzzy_score': combined_score,
            'individual_scores': individual_scores
        })
    
    if not results:
        return {
            'text': '',
            'confidence': 0,
            'method': 'easyocr_failed',
            'country': 'not visible',
            'material': 'not visible',
            'care': 'not visible',
            'num_detections': 0,
            'variants_tried': 0,
            'best_combined_score': 0
        }
    
    # Choose best result based on FUZZY SCORE
    best_result = max(results, key=lambda x: x['combined_fuzzy_score'])
    
    return {
        'text': best_result['text'],
        'confidence': best_result['confidence'],
        'method': f"easyocr_{best_result['method']}",
        'country': best_result['country'],
        'material': best_result['material'],
        'care': best_result['care'],
        'num_detections': len(best_result['text'].split()),
        'variants_tried': len(variants),
        'best_combined_score': best_result['combined_fuzzy_score']
    }

def compare_field(predicted, ground_truth):
    """Compare predicted vs ground truth for a single field"""
    if ground_truth is None or ground_truth == "not visible":
        if predicted == "not visible":
            return True, 1.0, "both_none"
        else:
            return False, 0.0, "no_ground_truth"
    
    if predicted is None or predicted == "not visible":
        return False, 0.0, "prediction_failed"
    
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    
    if pred_norm is None or gt_norm is None:
        return False, 0.0, "normalization_failed"
    
    exact = (pred_norm == gt_norm)
    fuzzy = fuzzy_match_score(predicted, ground_truth)
    
    if exact:
        status = "exact_match"
    elif fuzzy > 0.9:
        status = "near_match"
    elif fuzzy > 0.7:
        status = "partial_match"
    else:
        status = "mismatch"
    
    return exact, fuzzy, status

# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_ocr_accuracy(tag_images_folder='tag_images', 
                         cropped_folder='cropped_tags',
                         output_csv='ocr_evaluation_easyocr_dec1.csv',
                         use_preprocessing=True):
    """Evaluate EasyOCR accuracy with fuzzy-score-based preprocessing selection"""
        # ADD THIS DEBUG SECTION:
    import os
    print(f"\nDEBUG INFO:")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Looking for: {tag_images_folder}")
    print(f"  Absolute path: {Path(tag_images_folder).resolve()}")
    print(f"  Exists: {Path(tag_images_folder).exists()}")

    print("\n" + "="*60)
    print("EASYOCR EVALUATION")
    if use_preprocessing:
        print("WITH FUZZY-SCORE-BASED PREPROCESSING SELECTION")
    print("="*60)
    print("Loading EasyOCR reader...")
    
    # Initialize EasyOCR
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        print("✓ EasyOCR loaded successfully")
    except Exception as e:
        print(f"✗ Error loading EasyOCR: {e}")
        print("\nPlease install EasyOCR:")
        print("  pip install easyocr")
        return None
    
    # Get all images with ground truth
    tag_images_folder = Path(tag_images_folder)
    image_files = list(tag_images_folder.glob('*.JPG')) + list(tag_images_folder.glob('*.jpg'))
    
    images_with_gt = [f for f in image_files if (tag_images_folder / f"{f.stem}.txt").exists()]
    
    if not images_with_gt:
        print("No images with ground truth found!")
        return None
    
    # Check for cropped versions
    cropped_folder = Path(cropped_folder)
    if not cropped_folder.exists():
        print(f"\n✗ Cropped folder not found: {cropped_folder}")
        print("  Using original images only")
        cropped_available = set()
    else:
        cropped_available = set(f.name for f in cropped_folder.glob('*.JPG')) | \
                           set(f.name for f in cropped_folder.glob('*.jpg'))
    
    num_using_cropped = len([f for f in images_with_gt if f.name in cropped_available])
    num_using_original = len(images_with_gt) - num_using_cropped
    
    print(f"\n{'='*60}")
    print(f"DATASET INFO")
    print(f"{'='*60}")
    print(f"Images with ground truth: {len(images_with_gt)}")
    print(f"Using cropped versions: {num_using_cropped}")
    print(f"Using original versions: {num_using_original}")
    if use_preprocessing:
        print(f"Preprocessing variants per image: 5")
        print(f"Selection method: Highest combined fuzzy score")
    print(f"{'='*60}\n")
    
    results = []
    
    for img_path in tqdm(images_with_gt, desc="Processing images"):
        # Determine which image to use
        if img_path.name in cropped_available:
            image_to_process = cropped_folder / img_path.name
            image_source = "cropped"
        else:
            image_to_process = img_path
            image_source = "original"
        
        # Load image
        img = cv2.imread(str(image_to_process))
        if img is None:
            print(f"Could not read: {image_to_process}")
            continue
        
        # Resize if needed
        img = resize_if_needed(img)
        
        # Load ground truth
        txt_path = tag_images_folder / f"{img_path.stem}.txt"
        ground_truth = load_ground_truth(txt_path)
        
        # Run OCR
        try:
            if use_preprocessing:
                ocr_result = run_easyocr_ensemble(img, reader, ground_truth)
            else:
                text, conf = run_easyocr_single(img, reader)
                ocr_result = {
                    'text': text or '',
                    'confidence': conf,
                    'method': 'easyocr_original',
                    'country': extract_country(text or ''),
                    'material': extract_material(text or ''),
                    'care': extract_care(text or ''),
                    'num_detections': len(text.split()) if text else 0,
                    'variants_tried': 1,
                    'best_combined_score': 0
                }
        except Exception as e:
            print(f"\nOCR failed for {img_path.name}: {e}")
            ocr_result = {
                'text': '',
                'confidence': 0,
                'method': 'failed',
                'country': 'not visible',
                'material': 'not visible',
                'care': 'not visible',
                'num_detections': 0,
                'variants_tried': 0,
                'best_combined_score': 0
            }
        
        # Compare each field
        country_exact, country_fuzzy, country_status = compare_field(
            ocr_result['country'], ground_truth['country']
        )
        material_exact, material_fuzzy, material_status = compare_field(
            ocr_result['material'], ground_truth['material']
        )
        care_exact, care_fuzzy, care_status = compare_field(
            ocr_result['care'], ground_truth['care']
        )
        
        results.append({
            'image': img_path.name,
            'image_source': image_source,
            'ocr_method': ocr_result.get('method', 'unknown'),
            'num_detections': ocr_result.get('num_detections', 0),
            'variants_tried': ocr_result.get('variants_tried', 0),
            'best_combined_score': ocr_result.get('best_combined_score', 0),
            'country_pred': ocr_result['country'],
            'country_true': ground_truth['country'],
            'country_exact': country_exact,
            'country_fuzzy': country_fuzzy,
            'country_status': country_status,
            'material_pred': ocr_result['material'],
            'material_true': ground_truth['material'],
            'material_exact': material_exact,
            'material_fuzzy': material_fuzzy,
            'material_status': material_status,
            'care_pred': ocr_result['care'],
            'care_true': ground_truth['care'],
            'care_exact': care_exact,
            'care_fuzzy': care_fuzzy,
            'care_status': care_status,
            'ocr_confidence': ocr_result['confidence'],
            'full_text': ocr_result['text']
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS (EASYOCR)")
    print(f"{'='*60}")
    
    print(f"\nImage Sources:")
    print(f"  Cropped: {(df['image_source'] == 'cropped').sum()}")
    print(f"  Original: {(df['image_source'] == 'original').sum()}")
    
    if use_preprocessing:
        print(f"\nPreprocessing Methods Selected (by fuzzy score):")
        print(df['ocr_method'].value_counts().to_string())
        print(f"\nAverage combined fuzzy score: {df['best_combined_score'].mean():.3f}")
    
    for field in ['country', 'material', 'care']:
        valid_rows = df[df[f'{field}_true'].notna()]
        
        if len(valid_rows) > 0:
            exact_acc = valid_rows[f'{field}_exact'].sum() / len(valid_rows) * 100
            fuzzy_avg = valid_rows[f'{field}_fuzzy'].mean() * 100
            
            print(f"\n{field.upper()}:")
            print(f"  Total samples: {len(valid_rows)}")
            print(f"  Exact Match: {exact_acc:.1f}% ({valid_rows[f'{field}_exact'].sum()}/{len(valid_rows)})")
            print(f"  Avg Similarity: {fuzzy_avg:.1f}%")
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Detailed results saved to: {output_csv}")
    
    return df


if __name__ == "__main__":
    log_dir = "codecarbon_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name="easyocr_tag_eval",
        output_dir="codecarbon_logs",  # folder where logs will be saved
        log_level="info"
    )

    tracker.start()  # start measuring

    df = evaluate_ocr_accuracy(
        tag_images_folder='tag_images',
        cropped_folder='cropped_tags',
        output_csv='ocr_evaluation_easyocr_dec1_emissions.csv',
        use_preprocessing=True
    )

    # Stop measuring and get total emissions (in kg CO2eq)
    emissions_kg = tracker.stop()
    
    if df is not None:

        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Estimated carbon emissions for this run: {emissions_kg:.6f} kg CO₂e")
