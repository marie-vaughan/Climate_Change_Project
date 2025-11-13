import albumentations as A
import cv2
import os
from pathlib import Path
import easyocr
from tqdm import tqdm

# Create output directory
os.makedirs('augmented_tags', exist_ok=True)

# Define augmentation pipeline (similar to real-world conditions)
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.Rotate(limit=15, p=0.6),
    A.Perspective(scale=(0.05, 0.15), p=0.4),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.RandomShadow(p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
])

# Light augmentation (for closer to original)
light_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Rotate(limit=5, p=0.4),
])

def resize_if_needed(img, max_dimension=1920):
    """Resize image if too large"""
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

def augment_dataset(input_folder, output_folder, num_augmentations=20):
    """
    Generate augmented versions of all images in input folder
    
    Args:
        input_folder: Path to original tag images
        output_folder: Path to save augmented images
        num_augmentations: Number of augmented versions per image
    """
    # Get all image files
    image_files = list(Path(input_folder).glob('*.JPG'))
    
    print(f"Found {len(image_files)} images")
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Load image
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Could not load {img_path}")
            continue
        
        # Resize if needed
        img = resize_if_needed(img)
        
        # Save original (resized)
        original_name = img_path.stem
        cv2.imwrite(f'{output_folder}/{original_name}_original.jpg', img)
        
        # Generate augmented versions
        for i in range(num_augmentations):
            # Use heavy augmentation for most, light for some
            if i < num_augmentations * 0.7:
                augmented = transform(image=img)['image']
            else:
                augmented = light_transform(image=img)['image']
            
            output_path = f'{output_folder}/{original_name}_aug_{i:03d}.jpg'
            cv2.imwrite(output_path, augmented)
    
    total_images = len(image_files) * (num_augmentations + 1)
    print(f"\nGenerated {total_images} total images!")
    print(f"Original: {len(image_files)}")
    print(f"Augmented: {len(image_files) * num_augmentations}")

def test_ocr_on_dataset(folder_path, sample_size=5):
    """
    Test OCR on a sample of images to verify quality
    """
    print("\n=== Testing OCR on sample images ===")
    reader = easyocr.Reader(['en'])
    
    image_files = list(Path(folder_path).glob('*.jpg'))[:sample_size]
    
    for img_path in image_files:
        print(f"\n--- {img_path.name} ---")
        img = cv2.imread(str(img_path))
        
        try:
            result = reader.readtext(img, min_size=10)
            for (bbox, text, prob) in result:
                print(f"  {text} ({prob:.2f})")
        except Exception as e:
            print(f"  Error: {e}")

# Main execution
if __name__ == "__main__":
    INPUT_FOLDER = 'tag_images' 
    OUTPUT_FOLDER = 'augmented_tags'
    
    # Generate augmented dataset
    # This will create 33 * 20 = 660 augmented images + 33 originals = 693 total
    augment_dataset(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        num_augmentations=20  # Adjust this number
    )
    
    # Test OCR on a few samples
    test_ocr_on_dataset(OUTPUT_FOLDER, sample_size=10)