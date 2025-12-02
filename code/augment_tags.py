# augment_tags.py
import albumentations as A
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import shutil
from codecarbon import EmissionsTracker
import json
from datetime import datetime

# Define augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.Perspective(scale=(0.02, 0.08), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomShadow(p=0.2),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
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

def augment_dataset(input_folder, output_folder, num_augmentations=5):
    """
    Augment synthetic tag images with emissions tracking
    """
    
    os.makedirs(f'{output_folder}/images', exist_ok=True)
    os.makedirs(f'{output_folder}/labels', exist_ok=True)
    
    # Get all synthetic images
    image_folder = Path(input_folder) / 'images'
    label_folder = Path(input_folder) / 'labels'
    
    image_files = sorted(list(image_folder.glob('*.jpg')))
    
    # Initialize emissions tracker
    tracker = EmissionsTracker(
        project_name="tag_augmentation",
        output_dir="emissions",
        output_file="augmentation_emissions.csv"
    )
    
    tracker.start()
    start_time = datetime.now()
    
    print(f"\n{'='*60}")
    print(f"AUGMENTING SYNTHETIC TAGS")
    print(f"{'='*60}")
    print(f"Found {len(image_files)} images")
    print(f"Creating {num_augmentations} augmentations per image")
    print(f"Total output: {len(image_files) * (num_augmentations + 1)} images")
    print(f"Tracking emissions...")
    print(f"{'='*60}\n")
    
    for img_path in tqdm(image_files, desc="Augmenting"):
        # Load image
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Could not load {img_path}")
            continue
        
        # Resize if needed
        img = resize_if_needed(img)
        
        base_name = img_path.stem
        
        # Copy original image
        shutil.copy(img_path, f'{output_folder}/images/{base_name}_original.jpg')
        
        # Copy original label
        label_path = label_folder / f"{base_name}.txt"
        if label_path.exists():
            shutil.copy(label_path, f'{output_folder}/labels/{base_name}_original.txt')
        
        # Generate augmented versions
        for i in range(num_augmentations):
            augmented = transform(image=img)['image']
            
            aug_img_path = f'{output_folder}/images/{base_name}_aug_{i:02d}.jpg'
            aug_label_path = f'{output_folder}/labels/{base_name}_aug_{i:02d}.txt'
            
            cv2.imwrite(aug_img_path, augmented)
            
            # Copy label for augmented version
            if label_path.exists():
                shutil.copy(label_path, aug_label_path)
    
    # Stop tracking
    emissions = tracker.stop()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    total = len(image_files) * (num_augmentations + 1)
    
    # Save emissions report
    emissions_report = {
        'step': 'augmentation',
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration,
        'input_images': len(image_files),
        'output_images': total,
        'augmentations_per_image': num_augmentations,
        'emissions_kg_co2': emissions,
        'emissions_per_image_g_co2': (emissions * 1000) / total if total > 0 else 0,
        'energy_consumed_kwh': emissions / 0.475,
    }
    
    os.makedirs('emissions', exist_ok=True)
    with open('emissions/augmentation_report.json', 'w') as f:
        json.dump(emissions_report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Total images: {total}")
    print(f"  Output: {output_folder}/")
    print(f"\n{'='*60}")
    print(f"EMISSIONS REPORT")
    print(f"{'='*60}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"CO2 Emissions: {emissions*1000:.2f} g CO2")
    print(f"Per image: {(emissions*1000)/total:.4f} g CO2")
    print(f"Energy: {emissions/0.475:.4f} kWh")
    print(f"{'='*60}")

if __name__ == "__main__":
    augment_dataset(
        input_folder='synthetic_tags',
        output_folder='augmented_synthetic_tags',
        num_augmentations=5
    )