# run_complete_pipeline.py
import subprocess
import os

def run_pipeline():
    """Run the complete synthetic data generation pipeline with emissions tracking"""
    
    print("\n" + "="*60)
    print("COMPLETE SYNTHETIC TAG PIPELINE")
    print("WITH EMISSIONS TRACKING")
    print("="*60)
    
    # Step 1: Generate synthetic tags
    print("\n[Step 1/4] Generating 500 synthetic tags...")
    subprocess.run(['python', 'generate_synthetic_tagsnovtwentyfifth.py'])
    
    # Step 2: Augment 5x
    print("\n[Step 2/4] Augmenting 5x (500 → 3000 images)...")
    subprocess.run(['python', 'augment_tags.py'])
    
    # Step 3: Split into train/val/test
    print("\n[Step 3/4] Splitting into train/val/test...")
    subprocess.run(['python', 'split_dataset.py'])
    
    # Step 4: Summarize emissions
    print("\n[Step 4/4] Generating emissions summary...")
    subprocess.run(['python', 'summarize_emissions.py'])
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nDataset ready for EasyOCR fine-tuning:")
    print("  dataset_split/")
    print("    ├── train/ (2100 images)")
    print("    ├── val/ (450 images)")
    print("    └── test/ (450 images)")
    print("\nEmissions logs:")
    print("  emissions_logs/")
    print("    ├── generation_emissions.csv")
    print("    ├── augmentation_emissions.csv")
    print("    ├── split_emissions.csv")
    print("    ├── complete_summary.json")
    print("    └── emissions_breakdown.csv")

if __name__ == "__main__":
    run_pipeline()