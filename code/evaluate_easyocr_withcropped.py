import easyocr
import cv2
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher


def resize_if_needed(img, max_dimension=1920):
    """Resize image if too large"""
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

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
                ground_truth['country'] = None if value == "Not visible" else value
            elif line.startswith('MATERIAL:'):
                value = line.replace('MATERIAL:', '').strip()
                ground_truth['material'] = None if value == "Not visible" else value
            elif line.startswith('CARE:'):
                value = line.replace('CARE:', '').strip()
                ground_truth['care'] = None if value == "Not visible" else value
    
    return ground_truth

def extract_country(text):
    """Extract country from OCR text"""
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
            # Clean up
            country = re.split(r'\s+(?:LA|EN)', country)[0].strip()
            return country
    return None

def extract_material(text):
    """Extract material composition from OCR text"""
    # Pattern: Find anything that starts with X% and continues until we hit a stopping point
    pattern = r'\d+%\s*[A-Za-z]+(?:\s+[A-Za-z]+)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if matches:
        return ' '.join(matches)
    return None

def extract_care(text):
    """Extract care instructions from OCR text"""
    # Try to find care instructions section
    care_pattern = r'(?:HAND WASH|MACHINE WASH|DRY CLEAN|DO NOT|WASH|TUMBLE)(.*?)(?=MADE IN|MADE FOR|FABRIQUÉ|RN#|\d+%|$)'
    match = re.search(care_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        care_text = match.group(0).strip()
        # Remove French text
        lines = care_text.split('\n')
        english_lines = []
        
        french_indicators = ['À', 'É', 'LAVER', 'FROIDE', 'PAS', 'SÉCHER', 'REPASSAGE']
        
        for line in lines:
            if not any(indicator in line.upper() for indicator in french_indicators):
                if line.strip():
                    english_lines.append(line.strip())
        
        return ' '.join(english_lines) if english_lines else None
    
    return None

def extract_all_info(ocr_results):
    """Extract all information from OCR results"""
    # Combine text
    full_text = ' '.join([text for (bbox, text, prob) in ocr_results])
    
    # Also preserve structure
    sorted_results = sorted(ocr_results, key=lambda x: x[0][0][1])
    structured_text = '\n'.join([text for (bbox, text, prob) in sorted_results])
    
    country = extract_country(full_text) or extract_country(structured_text)
    material = extract_material(full_text) or extract_material(structured_text)
    care = extract_care(structured_text) or extract_care(full_text)
    
    return {
        'country': country,
        'material': material,
        'care': care,
        'full_text': full_text,
        'confidence': sum([prob for (_, _, prob) in ocr_results]) / len(ocr_results) if ocr_results else 0
    }

def normalize_text(text):
    """Normalize text for comparison"""
    if text is None:
        return None
    text = str(text).lower().strip()
    text = text.rstrip('.,;')
    text = re.sub(r'\s+', ' ', text)
    return text

def fuzzy_match_score(str1, str2):
    """Calculate similarity between two strings (0-1)"""
    if str1 is None or str2 is None:
        return 0.0
    str1_norm = normalize_text(str1)
    str2_norm = normalize_text(str2)
    return SequenceMatcher(None, str1_norm, str2_norm).ratio()

def compare_field(predicted, ground_truth):
    """Compare predicted vs ground truth for a single field"""
    if ground_truth is None and predicted is None:
        return True, 1.0, "both_none"
    if ground_truth is None:
        return False, 0.0, "no_ground_truth"
    if predicted is None:
        return False, 0.0, "prediction_failed"
    
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    
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


def evaluate_ocr_accuracy(tag_images_folder='tag_images', 
                         cropped_folder='cropped_tags',
                         output_csv='ocr_evaluation_cropped.csv'):
    """
    Evaluate OCR accuracy - prioritizes cropped images, falls back to originals
    
    Logic:
    1. For each image with ground truth in tag_images_folder
    2. Check if cropped version exists in cropped_folder
    3. Use cropped if available, otherwise use original
    4. Run OCR and evaluate against ground truth
    """
    
    # Initialize EasyOCR
    print("Loading EasyOCR model...")
    reader = easyocr.Reader(['en'])
    
    # Get all images with ground truth from tag_images
    tag_images_folder = Path(tag_images_folder)
    image_files = list(tag_images_folder.glob('*.JPG')) + list(tag_images_folder.glob('*.jpg'))
    
    # Filter to only images with ground truth
    images_with_gt = [f for f in image_files if (tag_images_folder / f"{f.stem}.txt").exists()]
    
    if not images_with_gt:
        print("No images with ground truth found!")
        return None
    
    # Check which have cropped versions
    cropped_folder = Path(cropped_folder)
    cropped_available = set(f.name for f in cropped_folder.glob('*.JPG')) | \
                       set(f.name for f in cropped_folder.glob('*.jpg'))
    
    num_using_cropped = len([f for f in images_with_gt if f.name in cropped_available])
    num_using_original = len(images_with_gt) - num_using_cropped
    
    print(f"\n{'='*60}")
    print(f"EVALUATING OCR ACCURACY")
    print(f"{'='*60}")
    print(f"Images with ground truth: {len(images_with_gt)}")
    print(f"Using cropped versions: {num_using_cropped}")
    print(f"Using original versions: {num_using_original}")
    print(f"{'='*60}\n")
    
    results = []
    
    for img_path in tqdm(images_with_gt, desc="Processing images"):
        # Determine which image to use
        if img_path.name in cropped_available:
            # Use cropped version
            image_to_process = cropped_folder / img_path.name
            image_source = "cropped"
        else:
            # Use original
            image_to_process = img_path
            image_source = "original"
        
        # Load image
        img = cv2.imread(str(image_to_process))
        if img is None:
            print(f"Could not read: {image_to_process}")
            continue
        
        # Resize if needed
        img = resize_if_needed(img)
        
        # Run OCR
        try:
            ocr_result = reader.readtext(img, min_size=10, text_threshold=0.7)
            prediction = extract_all_info(ocr_result)
        except Exception as e:
            print(f"OCR failed for {img_path.name}: {e}")
            prediction = {
                'country': None,
                'material': None,
                'care': None,
                'full_text': '',
                'confidence': 0
            }
        
        # Load ground truth (always from tag_images folder)
        txt_path = tag_images_folder / f"{img_path.stem}.txt"
        ground_truth = load_ground_truth(txt_path)
        
        # Compare each field
        country_exact, country_fuzzy, country_status = compare_field(
            prediction['country'], ground_truth['country']
        )
        material_exact, material_fuzzy, material_status = compare_field(
            prediction['material'], ground_truth['material']
        )
        care_exact, care_fuzzy, care_status = compare_field(
            prediction['care'], ground_truth['care']
        )
        
        results.append({
            'image': img_path.name,
            'image_source': image_source,
            'country_pred': prediction['country'],
            'country_true': ground_truth['country'],
            'country_exact': country_exact,
            'country_fuzzy': country_fuzzy,
            'country_status': country_status,
            'material_pred': prediction['material'],
            'material_true': ground_truth['material'],
            'material_exact': material_exact,
            'material_fuzzy': material_fuzzy,
            'material_status': material_status,
            'care_pred': prediction['care'],
            'care_true': ground_truth['care'],
            'care_exact': care_exact,
            'care_fuzzy': care_fuzzy,
            'care_status': care_status,
            'ocr_confidence': prediction['confidence'],
            'full_text': prediction['full_text']
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Overall breakdown
    print(f"\nImage Sources:")
    print(f"  Cropped: {(df['image_source'] == 'cropped').sum()}")
    print(f"  Original: {(df['image_source'] == 'original').sum()}")
    
    for field in ['country', 'material', 'care']:
        exact_col = f'{field}_exact'
        fuzzy_col = f'{field}_fuzzy'
        status_col = f'{field}_status'
        
        valid_rows = df[df[f'{field}_true'].notna()]
        
        if len(valid_rows) > 0:
            exact_acc = valid_rows[exact_col].sum() / len(valid_rows) * 100
            fuzzy_avg = valid_rows[fuzzy_col].mean() * 100
            
            status_counts = valid_rows[status_col].value_counts()
            
            print(f"\n{field.upper()}:")
            print(f"  Total samples: {len(valid_rows)}")
            print(f"  Exact Match: {exact_acc:.1f}% ({valid_rows[exact_col].sum()}/{len(valid_rows)})")
            print(f"  Avg Similarity: {fuzzy_avg:.1f}%")
            print(f"  Status breakdown:")
            for status, count in status_counts.items():
                pct = count / len(valid_rows) * 100
                print(f"    - {status}: {count} ({pct:.1f}%)")
    
    # Compare cropped vs original performance
    print(f"\n{'='*60}")
    print("CROPPED vs ORIGINAL COMPARISON")
    print(f"{'='*60}")
    
    cropped_df = df[df['image_source'] == 'cropped']
    original_df = df[df['image_source'] == 'original']
    
    if len(cropped_df) > 0 and len(original_df) > 0:
        for field in ['country', 'material', 'care']:
            cropped_valid = cropped_df[cropped_df[f'{field}_true'].notna()]
            original_valid = original_df[original_df[f'{field}_true'].notna()]
            
            if len(cropped_valid) > 0 and len(original_valid) > 0:
                cropped_acc = cropped_valid[f'{field}_exact'].sum() / len(cropped_valid) * 100
                original_acc = original_valid[f'{field}_exact'].sum() / len(original_valid) * 100
                
                print(f"\n{field.upper()}:")
                print(f"  Cropped: {cropped_acc:.1f}% ({len(cropped_valid)} samples)")
                print(f"  Original: {original_acc:.1f}% ({len(original_valid)} samples)")
                print(f"  Improvement: {cropped_acc - original_acc:+.1f}%")
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Total images processed: {len(df)}")
    print(f"Average OCR confidence: {df['ocr_confidence'].mean():.2f}")
    
    # Show worst mismatches
    print(f"\n{'='*60}")
    print("WORST MISMATCHES (Top 10)")
    print(f"{'='*60}")
    
    # Calculate overall error score for each image
    df['error_score'] = 0
    for field in ['country', 'material', 'care']:
        df['error_score'] += (1 - df[f'{field}_fuzzy'])
    
    worst = df.nlargest(10, 'error_score')
    
    for _, row in worst.iterrows():
        print(f"\n{row['image']} ({row['image_source']}) - error score: {row['error_score']:.2f}:")
        
        if row['country_status'] not in ['exact_match', 'both_none', 'no_ground_truth']:
            print(f"  COUNTRY ({row['country_fuzzy']:.0%} match):")
            print(f"    Expected: '{row['country_true']}'")
            print(f"    Got:      '{row['country_pred']}'")
        
        if row['material_status'] not in ['exact_match', 'both_none', 'no_ground_truth']:
            print(f"  MATERIAL ({row['material_fuzzy']:.0%} match):")
            print(f"    Expected: '{row['material_true']}'")
            print(f"    Got:      '{row['material_pred']}'")
        
        if row['care_status'] not in ['exact_match', 'both_none', 'no_ground_truth']:
            print(f"  CARE ({row['care_fuzzy']:.0%} match):")
            print(f"    Expected: '{row['care_true'][:50]}...'")
            print(f"    Got:      '{row['care_pred'][:50] if row['care_pred'] else None}...'")
    
    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Detailed results saved to: {output_csv}")
    
    return df


if __name__ == "__main__":
    # Run evaluation
    df = evaluate_ocr_accuracy(
        tag_images_folder='tag_images',     # Where ground truth .txt files are
        cropped_folder='cropped_tags',       # Where cropped images are
        output_csv='ocr_evaluation.csv'
    )
    
    # Additional analysis
    if df is not None:
        print(f"\n{'='*60}")
        print("SAVE LOCATIONS")
        print(f"{'='*60}")
        print(f"Full results: ocr_evaluation.csv")
        print(f"Open in Excel/pandas for detailed analysis")


# 1. **Prioritizes cropped images:**
#    - Checks if `IMG_001.JPG` exists in `cropped_tags/`
#    - If yes → use cropped version
#    - If no → use original from `tag_images/`

# 2. **Tracks image source:**
#    - New column `image_source` shows whether "cropped" or "original" was used

# 3. **Compares performance:**
#    - Shows accuracy for cropped vs original images
#    - Calculates improvement from cropping

# 4. **Ground truth location:**
#    - Always reads `.txt` files from `tag_images/` folder
#    - Works even if you don't copy `.txt` to `cropped_tags/`

