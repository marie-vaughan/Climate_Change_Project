# split_dataset.py
import os
import shutil
from pathlib import Path
import random
from codecarbon import EmissionsTracker
import json
from datetime import datetime

def split_dataset(input_folder, output_folder='dataset_split', 
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split augmented dataset into train/val/test with emissions tracking
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    # Initialize emissions tracker
    tracker = EmissionsTracker(
        project_name="dataset_split",
        output_dir="emissions",
        output_file="split_emissions.csv"
    )
    
    tracker.start()
    start_time = datetime.now()
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{output_folder}/{split}/images', exist_ok=True)
        os.makedirs(f'{output_folder}/{split}/labels', exist_ok=True)
    
    # Get all image files
    image_folder = Path(input_folder) / 'images'
    label_folder = Path(input_folder) / 'labels'
    
    image_files = sorted(list(image_folder.glob('*.jpg')))
    
    # Group by base name
    base_names = {}
    for img_path in image_files:
        name = img_path.stem
        if '_original' in name:
            base = name.replace('_original', '')
        elif '_aug_' in name:
            base = name.split('_aug_')[0]
        else:
            base = name
        
        if base not in base_names:
            base_names[base] = []
        base_names[base].append(img_path)
    
    # Shuffle base names
    base_list = list(base_names.keys())
    random.shuffle(base_list)
    
    # Calculate split indices
    total = len(base_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split
    train_bases = base_list[:train_end]
    val_bases = base_list[train_end:val_end]
    test_bases = base_list[val_end:]
    
    print(f"\n{'='*60}")
    print(f"SPLITTING DATASET")
    print(f"{'='*60}")
    print(f"Total base images: {total}")
    print(f"Train: {len(train_bases)} ({len(train_bases)/total*100:.1f}%)")
    print(f"Val:   {len(val_bases)} ({len(val_bases)/total*100:.1f}%)")
    print(f"Test:  {len(test_bases)} ({len(test_bases)/total*100:.1f}%)")
    print(f"Tracking emissions...")
    print(f"{'='*60}\n")
    
    # Copy files to respective splits
    def copy_to_split(bases, split_name):
        count = 0
        for base in bases:
            for img_path in base_names[base]:
                # Copy image
                shutil.copy(img_path, f'{output_folder}/{split_name}/images/')
                
                # Copy label
                label_path = label_folder / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy(label_path, f'{output_folder}/{split_name}/labels/')
                
                count += 1
        return count
    
    train_count = copy_to_split(train_bases, 'train')
    val_count = copy_to_split(val_bases, 'val')
    test_count = copy_to_split(test_bases, 'test')
    
    # Stop tracking
    emissions = tracker.stop()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    total_files = train_count + val_count + test_count
    
    # Save emissions report
    emissions_report = {
        'step': 'dataset_split',
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration,
        'total_images': total_files,
        'train_images': train_count,
        'val_images': val_count,
        'test_images': test_count,
        'emissions_kg_co2': emissions,
        'emissions_per_image_g_co2': (emissions * 1000) / total_files if total_files > 0 else 0,
        'energy_consumed_kwh': emissions / 0.475,
    }
    
    os.makedirs('emissions', exist_ok=True)
    with open('emissions/split_report.json', 'w') as f:
        json.dump(emissions_report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DATASET SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Train: {train_count} images")
    print(f"✓ Val:   {val_count} images")
    print(f"✓ Test:  {test_count} images")
    print(f"✓ Total: {total_files} images")
    print(f"\n{'='*60}")
    print(f"EMISSIONS REPORT")
    print(f"{'='*60}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"CO2 Emissions: {emissions*1000:.2f} g CO2")
    print(f"Per image: {(emissions*1000)/total_files:.4f} g CO2")
    print(f"Energy: {emissions/0.475:.4f} kWh")
    print(f"{'='*60}")

if __name__ == "__main__":
    split_dataset(
        input_folder='augmented_synthetic_tags',
        output_folder='dataset_split',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )