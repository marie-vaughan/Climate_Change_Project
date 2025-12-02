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

    
#     print(f"\n{'='*60}")
#     print("OCR RESULTS COMPARISON")
#     print(f"{'='*60}")
#     print(f"CSV 1: {csv1}")
#     print(f"CSV 2: {csv2}")
#     print(f"Fuzzy match threshold: {fuzzy_threshold*100}%")
#     print(f"{'='*60}\n")
    
#     # Calculate accuracy metrics for each field
#     def calc_accuracy(df, field):
#         valid = df[df[f'{field}_true'].notna()]
#         if len(valid) == 0:
#             return {
#                 'exact': 0,
#                 'fuzzy_avg': 0,
#                 'fuzzy_acceptable': 0,
#                 'near_match': 0,
#                 'partial_match': 0,
#                 'total': 0
#             }
        
#         exact = valid[f'{field}_exact'].sum() / len(valid) * 100
#         fuzzy_avg = valid[f'{field}_fuzzy'].mean() * 100
        
#         # Count by status
#         fuzzy_acceptable = (valid[f'{field}_fuzzy'] >= fuzzy_threshold).sum() / len(valid) * 100
#         near_match = (valid[f'{field}_status'] == 'near_match').sum()
#         partial_match = (valid[f'{field}_status'] == 'partial_match').sum()
        
#         return {
#             'exact': exact,
#             'fuzzy_avg': fuzzy_avg,
#             'fuzzy_acceptable': fuzzy_acceptable,
#             'near_match': near_match,
#             'partial_match': partial_match,
#             'total': len(valid)
#         }
    
#     # Compare overall accuracy with fuzzy matches
#     print("OVERALL ACCURACY COMPARISON")
#     print(f"{'='*60}")
    
#     comparison_data = []
    
#     for field in ['country', 'material', 'care']:
#         stats1 = calc_accuracy(df1, field)
#         stats2 = calc_accuracy(df2, field)
        
#         improvement_exact = stats2['exact'] - stats1['exact']
#         improvement_fuzzy = stats2['fuzzy_avg'] - stats1['fuzzy_avg']
#         improvement_acceptable = stats2['fuzzy_acceptable'] - stats1['fuzzy_acceptable']
        
#         comparison_data.append({
#             'field': field,
#             'csv1_exact': stats1['exact'],
#             'csv2_exact': stats2['exact'],
#             'improvement_exact': improvement_exact,
#             'csv1_fuzzy_avg': stats1['fuzzy_avg'],
#             'csv2_fuzzy_avg': stats2['fuzzy_avg'],
#             'improvement_fuzzy': improvement_fuzzy,
#             'csv1_fuzzy_acceptable': stats1['fuzzy_acceptable'],
#             'csv2_fuzzy_acceptable': stats2['fuzzy_acceptable'],
#             'improvement_acceptable': improvement_acceptable,
#             'csv1_near_match': stats1['near_match'],
#             'csv2_near_match': stats2['near_match'],
#             'csv1_partial_match': stats1['partial_match'],
#             'csv2_partial_match': stats2['partial_match'],
#         })
        
#         print(f"\n{field.upper()}:")
#         print(f"  CSV 1:")
#         print(f"    - Exact Match: {stats1['exact']:.1f}%")
#         print(f"    - Avg Similarity: {stats1['fuzzy_avg']:.1f}%")
#         print(f"    - Acceptable (≥{fuzzy_threshold*100}%): {stats1['fuzzy_acceptable']:.1f}%")
#         print(f"    - Near matches (>90%): {stats1['near_match']}")
#         print(f"    - Partial matches (70-90%): {stats1['partial_match']}")
        
#         print(f"  CSV 2:")
#         print(f"    - Exact Match: {stats2['exact']:.1f}%")
#         print(f"    - Avg Similarity: {stats2['fuzzy_avg']:.1f}%")
#         print(f"    - Acceptable (≥{fuzzy_threshold*100}%): {stats2['fuzzy_acceptable']:.1f}%")
#         print(f"    - Near matches (>90%): {stats2['near_match']}")
#         print(f"    - Partial matches (70-90%): {stats2['partial_match']}")
        
#         print(f"  Improvement:")
#         print(f"    - Exact: {improvement_exact:+.1f}%")
#         print(f"    - Fuzzy Avg: {improvement_fuzzy:+.1f}%")
#         print(f"    - Acceptable: {improvement_acceptable:+.1f}%")
    
#     # If CSV2 has image_source column, show cropped vs original breakdown
#     if 'image_source' in df2.columns:
#         print(f"\n{'='*60}")
#         print("CSV 2 BREAKDOWN (Cropped vs Original)")
#         print(f"{'='*60}")
        
#         cropped_df = df2[df2['image_source'] == 'cropped']
#         original_df = df2[df2['image_source'] == 'original']
        
#         print(f"Images using cropped: {len(cropped_df)}")
#         print(f"Images using original: {len(original_df)}")
        
#         if len(cropped_df) > 0 and len(original_df) > 0:
#             for field in ['country', 'material', 'care']:
#                 cropped_stats = calc_accuracy(cropped_df, field)
#                 original_stats = calc_accuracy(original_df, field)
                
#                 print(f"\n{field.upper()}:")
#                 print(f"  Cropped:")
#                 print(f"    - Exact: {cropped_stats['exact']:.1f}%")
#                 print(f"    - Fuzzy Avg: {cropped_stats['fuzzy_avg']:.1f}%")
#                 print(f"    - Acceptable: {cropped_stats['fuzzy_acceptable']:.1f}%")
                
#                 print(f"  Original:")
#                 print(f"    - Exact: {original_stats['exact']:.1f}%")
#                 print(f"    - Fuzzy Avg: {original_stats['fuzzy_avg']:.1f}%")
#                 print(f"    - Acceptable: {original_stats['fuzzy_acceptable']:.1f}%")
                
#                 print(f"  Cropping benefit:")
#                 print(f"    - Exact: {cropped_stats['exact'] - original_stats['exact']:+.1f}%")
#                 print(f"    - Fuzzy: {cropped_stats['fuzzy_avg'] - original_stats['fuzzy_avg']:+.1f}%")
    
#     # Per-image comparison with fuzzy scores
#     print(f"\n{'='*60}")
#     print("PER-IMAGE COMPARISON")
#     print(f"{'='*60}")
    
#     # Merge on image name
#     merged = df1.merge(df2, on='image', suffixes=('_csv1', '_csv2'))
    
#     # Calculate scores for each image
#     improved_images = []
#     worsened_images = []
#     fuzzy_improved = []
    
#     for _, row in merged.iterrows():
#         # Exact match score (0-3, one point per field)
#         csv1_exact_score = (row['country_exact_csv1'] + 
#                            row['material_exact_csv1'] + 
#                            row['care_exact_csv1'])
#         csv2_exact_score = (row['country_exact_csv2'] + 
#                            row['material_exact_csv2'] + 
#                            row['care_exact_csv2'])
        
#         # Fuzzy match score (average similarity)
#         csv1_fuzzy_score = (row['country_fuzzy_csv1'] + 
#                            row['material_fuzzy_csv1'] + 
#                            row['care_fuzzy_csv1']) / 3
#         csv2_fuzzy_score = (row['country_fuzzy_csv2'] + 
#                            row['material_fuzzy_csv2'] + 
#                            row['care_fuzzy_csv2']) / 3
        
#         exact_diff = csv2_exact_score - csv1_exact_score
#         fuzzy_diff = csv2_fuzzy_score - csv1_fuzzy_score
        
#         if exact_diff > 0:
#             improved_images.append({
#                 'image': row['image'],
#                 'csv1_exact': csv1_exact_score,
#                 'csv2_exact': csv2_exact_score,
#                 'improvement': exact_diff,
#                 'csv1_fuzzy': csv1_fuzzy_score,
#                 'csv2_fuzzy': csv2_fuzzy_score,
#                 'fuzzy_improvement': fuzzy_diff
#             })
#         elif exact_diff < 0:
#             worsened_images.append({
#                 'image': row['image'],
#                 'csv1_exact': csv1_exact_score,
#                 'csv2_exact': csv2_exact_score,
#                 'decline': exact_diff,
#                 'csv1_fuzzy': csv1_fuzzy_score,
#                 'csv2_fuzzy': csv2_fuzzy_score,
#                 'fuzzy_decline': fuzzy_diff
#             })
        
#         # Track images that improved on fuzzy score even if exact stayed same
#         if exact_diff == 0 and fuzzy_diff > 0.05:  # At least 5% improvement
#             fuzzy_improved.append({
#                 'image': row['image'],
#                 'csv1_fuzzy': csv1_fuzzy_score,
#                 'csv2_fuzzy': csv2_fuzzy_score,
#                 'fuzzy_improvement': fuzzy_diff
#             })
    
#     print(f"\nExact Match Changes:")
#     print(f"  Images that improved: {len(improved_images)}")
#     print(f"  Images that worsened: {len(worsened_images)}")
#     print(f"  Images unchanged: {len(merged) - len(improved_images) - len(worsened_images)}")
    
#     print(f"\nFuzzy Match Analysis:")
#     print(f"  Images with improved fuzzy score (but same exact): {len(fuzzy_improved)}")
    
#     if improved_images:
#         print(f"\nTop 10 Most Improved (Exact Match):")
#         improved_df = pd.DataFrame(improved_images).nlargest(10, 'improvement')
#         for _, row in improved_df.iterrows():
#             print(f"  {row['image']}:")
#             print(f"    Exact: {row['csv1_exact']:.0f} → {row['csv2_exact']:.0f} (+{row['improvement']:.0f})")
#             print(f"    Fuzzy: {row['csv1_fuzzy']:.2%} → {row['csv2_fuzzy']:.2%} ({row['fuzzy_improvement']:+.2%})")
    
#     if worsened_images:
#         print(f"\nTop 10 Most Worsened (Exact Match):")
#         worsened_df = pd.DataFrame(worsened_images).nsmallest(10, 'decline')
#         for _, row in worsened_df.iterrows():
#             print(f"  {row['image']}:")
#             print(f"    Exact: {row['csv1_exact']:.0f} → {row['csv2_exact']:.0f} ({row['decline']:.0f})")
#             print(f"    Fuzzy: {row['csv1_fuzzy']:.2%} → {row['csv2_fuzzy']:.2%} ({row['fuzzy_decline']:+.2%})")
    
#     if fuzzy_improved:
#         print(f"\nTop 10 Fuzzy Score Improvements (Same Exact Match):")
#         fuzzy_df = pd.DataFrame(fuzzy_improved).nlargest(10, 'fuzzy_improvement')
#         for _, row in fuzzy_df.iterrows():
#             print(f"  {row['image']}:")
#             print(f"    Fuzzy: {row['csv1_fuzzy']:.2%} → {row['csv2_fuzzy']:.2%} (+{row['fuzzy_improvement']:.2%})")
    
#     # Analyze match quality distribution changes
#     print(f"\n{'='*60}")
#     print("MATCH QUALITY DISTRIBUTION")
#     print(f"{'='*60}")
    
#     for field in ['country', 'material', 'care']:
#         print(f"\n{field.upper()}:")
        
#         # CSV 1 distribution
#         csv1_valid = df1[df1[f'{field}_true'].notna()]
#         csv1_status = csv1_valid[f'{field}_status'].value_counts()
        
#         # CSV 2 distribution
#         csv2_valid = df2[df2[f'{field}_true'].notna()]
#         csv2_status = csv2_valid[f'{field}_status'].value_counts()
        
#         print(f"  CSV 1:")
#         for status in ['exact_match', 'near_match', 'partial_match', 'mismatch']:
#             count = csv1_status.get(status, 0)
#             pct = count / len(csv1_valid) * 100 if len(csv1_valid) > 0 else 0
#             print(f"    - {status}: {count} ({pct:.1f}%)")
        
#         print(f"  CSV 2:")
#         for status in ['exact_match', 'near_match', 'partial_match', 'mismatch']:
#             count = csv2_status.get(status, 0)
#             pct = count / len(csv2_valid) * 100 if len(csv2_valid) > 0 else 0
#             print(f"    - {status}: {count} ({pct:.1f}%)")
    
#     # Save comparison
#     comparison_df = pd.DataFrame(comparison_data)
#     comparison_df.to_csv(output_comparison, index=False)
#     print(f"\n✓ Comparison saved to: {output_comparison}")
    
#     # Create visualizations
#     create_comparison_charts(comparison_df, df1, df2)
    
#     return comparison_df, merged

# def create_comparison_charts(comparison_df, df1, df2):
#     """Create comprehensive visualization comparing the two results"""
    
#     fig = plt.figure(figsize=(18, 10))
#     gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
#     # 1. Exact match comparison
#     ax1 = fig.add_subplot(gs[0, 0])
#     x = np.arange(len(comparison_df))
#     width = 0.35
    
#     ax1.bar(x - width/2, comparison_df['csv1_exact'], width, 
#            label='CSV 1', color='#3b82f6', alpha=0.8)
#     ax1.bar(x + width/2, comparison_df['csv2_exact'], width, 
#            label='CSV 2', color='#10b981', alpha=0.8)
    
#     ax1.set_xlabel('Field', fontsize=11)
#     ax1.set_ylabel('Exact Match Accuracy (%)', fontsize=11)
#     ax1.set_title('Exact Match Comparison', fontsize=12, fontweight='bold')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(comparison_df['field'].str.title())
#     ax1.legend()
#     ax1.grid(axis='y', alpha=0.3)
    
#     # 2. Fuzzy average comparison
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax2.bar(x - width/2, comparison_df['csv1_fuzzy_avg'], width, 
#            label='CSV 1', color='#3b82f6', alpha=0.8)
#     ax2.bar(x + width/2, comparison_df['csv2_fuzzy_avg'], width, 
#            label='CSV 2', color='#10b981', alpha=0.8)
    
#     ax2.set_xlabel('Field', fontsize=11)
#     ax2.set_ylabel('Average Similarity (%)', fontsize=11)
#     ax2.set_title('Fuzzy Match Comparison', fontsize=12, fontweight='bold')
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(comparison_df['field'].str.title())
#     ax2.legend()
#     ax2.grid(axis='y', alpha=0.3)
    
#     # 3. Acceptable fuzzy matches
#     ax3 = fig.add_subplot(gs[0, 2])
#     ax3.bar(x - width/2, comparison_df['csv1_fuzzy_acceptable'], width, 
#            label='CSV 1', color='#3b82f6', alpha=0.8)
#     ax3.bar(x + width/2, comparison_df['csv2_fuzzy_acceptable'], width, 
#            label='CSV 2', color='#10b981', alpha=0.8)
    
#     ax3.set_xlabel('Field', fontsize=11)
#     ax3.set_ylabel('Acceptable Matches (%)', fontsize=11)
#     ax3.set_title('Acceptable Fuzzy Matches (≥70%)', fontsize=12, fontweight='bold')
#     ax3.set_xticks(x)
#     ax3.set_xticklabels(comparison_df['field'].str.title())
#     ax3.legend()
#     ax3.grid(axis='y', alpha=0.3)
    
#     # 4. Improvement in exact match
#     ax4 = fig.add_subplot(gs[1, 0])
#     colors = ['#10b981' if x > 0 else '#ef4444' for x in comparison_df['improvement_exact']]
#     ax4.bar(x, comparison_df['improvement_exact'], color=colors, alpha=0.8)
    
#     ax4.set_xlabel('Field', fontsize=11)
#     ax4.set_ylabel('Improvement (%)', fontsize=11)
#     ax4.set_title('Exact Match Improvement', fontsize=12, fontweight='bold')
#     ax4.set_xticks(x)
#     ax4.set_xticklabels(comparison_df['field'].str.title())
#     ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
#     ax4.grid(axis='y', alpha=0.3)
    
#     # 5. Improvement in fuzzy match
#     ax5 = fig.add_subplot(gs[1, 1])
#     colors = ['#10b981' if x > 0 else '#ef4444' for x in comparison_df['improvement_fuzzy']]
#     ax5.bar(x, comparison_df['improvement_fuzzy'], color=colors, alpha=0.8)
    
#     ax5.set_xlabel('Field', fontsize=11)
#     ax5.set_ylabel('Improvement (%)', fontsize=11)
#     ax5.set_title('Fuzzy Match Improvement', fontsize=12, fontweight='bold')
#     ax5.set_xticks(x)
#     ax5.set_xticklabels(comparison_df['field'].str.title())
#     ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
#     ax5.grid(axis='y', alpha=0.3)
    
#     # 6. Match quality distribution stacked bar
#     ax6 = fig.add_subplot(gs[1, 2])
    
#     fields = comparison_df['field'].tolist()
#     exact = comparison_df['csv2_exact'].tolist()
#     near = [comparison_df.loc[i, 'csv2_near_match'] for i in range(len(comparison_df))]
#     partial = [comparison_df.loc[i, 'csv2_partial_match'] for i in range(len(comparison_df))]
    
#     # Convert counts to percentages (approximate)
#     # This is simplified - you'd need total counts for accurate percentages
    
#     ax6.bar(x, exact, label='Exact (100%)', color='#10b981', alpha=0.8)
#     # Add text labels showing improvement
#     for i, (field, imp) in enumerate(zip(fields, comparison_df['improvement_exact'])):
#         if imp != 0:
#             ax6.text(i, exact[i] + 2, f'{imp:+.1f}%', 
#                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
#     ax6.set_xlabel('Field', fontsize=11)
#     ax6.set_ylabel('Exact Match (%)', fontsize=11)
#     ax6.set_title('CSV 2 Quality with Improvements', fontsize=12, fontweight='bold')
#     ax6.set_xticks(x)
#     ax6.set_xticklabels([f.title() for f in fields])
#     ax6.legend()
#     ax6.grid(axis='y', alpha=0.3)
    
#     plt.savefig('ocr_comparison_detailed.png', dpi=300, bbox_inches='tight')
#     print(f"\n✓ Detailed charts saved to: ocr_comparison_detailed.png")
#     plt.show()

# # ============================================================================
# # USAGE
# # ============================================================================

if __name__ == "__main__":
    # Compare two evaluation results
    comparison_df, merged_df = compare_ocr_results(
        csv1='ocr_evaluation_initial.csv',      # Your first run
        csv2='ocr_evaluation_cropped.csv',  # Your second run
        output_comparison='ocr_comparison.csv',
        fuzzy_threshold=0.2  # Consider ≥20% similarity as "acceptable"
    )
