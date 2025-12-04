import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_ocr_results(csv1='ocr_evaluation_initial.csv', 
                       csv2='ocr_evaluation_cropped.csv',
                       output_comparison='ocr_comparison.csv',
                       fuzzy_threshold=0.2):
    """
    Compare two OCR evaluation results with fuzzy match analysis
    
    Args:
        csv1: First evaluation CSV (e.g., original images only)
        csv2: Second evaluation CSV (e.g., with cropped images)
        output_comparison: Where to save the comparison
        fuzzy_threshold: Threshold for "acceptable" fuzzy match (default: 0.7 = 70%)
    """
    
    # Load both CSVs
    original = pd.read_csv(csv1)
    cropped = pd.read_csv(csv2)

    # graph counts of country_fuzzy greater than 0
   
    plt.figure(figsize=(10, 6))
    sns.histplot(original[original['country_fuzzy'] > 0]['country_fuzzy'], bins=20, color='blue', alpha=0.5, label='Original')
    sns.histplot(cropped[cropped['country_fuzzy'] > 0]['country_fuzzy'], bins=20, color='orange', alpha=0.5, label='Cropped')
    plt.title('Distribution of Country Fuzzy Match Scores')
    plt.xlabel('Fuzzy Match Score')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

#calculate percent of country_fuzzy greater than 0.2
    original_above_threshold = (original['country_fuzzy'] > fuzzy_threshold).sum() / len(original) * 100
    cropped_above_threshold = (cropped['country_fuzzy'] > fuzzy_threshold).sum() / len(cropped) * 100
    print(f"Original images - % of country_fuzzy > {fuzzy_threshold}: {original_above_threshold:.2f}%")
    print(f"Cropped images - % of country_fuzzy > {fuzzy_threshold}: {cropped_above_threshold:.2f}%")

    

if __name__ == "__main__":
    # Compare two evaluation results
    comparison_df, merged_df = compare_ocr_results(
        csv1='ocr_evaluation_initial.csv',      # Your first run
        csv2='ocr_evaluation_cropped.csv',  # Your second run
        output_comparison='ocr_comparison.csv',
        fuzzy_threshold=0.2  # Consider ≥20% similarity as "acceptable"
    )
