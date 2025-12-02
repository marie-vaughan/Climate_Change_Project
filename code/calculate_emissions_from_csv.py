"""
Calculate CO2 emissions for each row in the OCR evaluation CSV
using the predicted country, material, and care values.
"""

import pandas as pd
import json
from pathlib import Path
from parser import parse_from_text
from calc import estimate
from tqdm import tqdm

def create_text_from_predictions(row):
    """Create a text string from predicted values"""
    parts = []
    
    # Add country if available
    if pd.notna(row['country_pred']) and row['country_pred'] != 'not visible':
        parts.append(f"MADE IN {row['country_pred']}")
    
    # Add material if available
    if pd.notna(row['material_pred']) and row['material_pred'] != 'not visible':
        parts.append(row['material_pred'])
    
    # Add care if available
    if pd.notna(row['care_pred']) and row['care_pred'] != 'not visible':
        parts.append(row['care_pred'])
    
    return " ".join(parts)

def calculate_emissions_for_row(row, default_weight_g=1000, washes_per_month=2.0):
    """Calculate CO2 emissions for a single row"""
    
    # Create text from predictions
    text = create_text_from_predictions(row)
    
    if not text or text.strip() == '':
        return {
            'total_kgco2e': None,
            'materials_kgco2e': None,
            'manufacturing_kgco2e': None,
            'washing_kgco2e': None,
            'parsed_country': None,
            'parsed_materials': None,
            'calculation_error': 'No valid predictions'
        }
    
    try:
        # Parse the text
        record = parse_from_text(
            text,
            garment_type=None,
            default_weight_g=default_weight_g,
            washes_per_month=washes_per_month
        )
        
        # Calculate emissions
        result = estimate(record, preferred_mode="ship")
        
        return {
            'total_kgco2e': round(result.total_kgco2e, 3),
            'materials_kgco2e': round(result.breakdown.get('materials', 0), 3),
            'manufacturing_kgco2e': round(result.breakdown.get('manufacturing', 0), 3),
            'washing_kgco2e': round(result.breakdown.get('washing', 0), 3),
            'parsed_country': record.origin_country,
            'parsed_materials': [{"fiber": m.fiber, "pct": m.pct} for m in record.materials] if record.materials else [],
            'calculation_error': None
        }
    
    except Exception as e:
        return {
            'total_kgco2e': None,
            'materials_kgco2e': None,
            'manufacturing_kgco2e': None,
            'washing_kgco2e': None,
            'parsed_country': None,
            'parsed_materials': None,
            'calculation_error': str(e)
        }

def process_ocr_evaluation_csv(
    input_csv='ocr_evaluation_easyocr_dec1.csv',
    output_csv='emissions_estimates_from_ocr.csv',
    default_weight_g=1000,
    washes_per_month=2.0
):
    """
    Process the OCR evaluation CSV and calculate emissions for each row
    """
    
    print("="*80)
    print("CALCULATING CO2 EMISSIONS FROM OCR PREDICTIONS")
    print("="*80)
    
    # Read the CSV
    print(f"\nReading: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Found {len(df)} rows")
    print(f"Using default weight: {default_weight_g}g ({default_weight_g/1000}kg)")
    print(f"Using washes per month: {washes_per_month}")
    
    # Calculate emissions for each row
    print("\nCalculating emissions...")
    
    emissions_results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        result = calculate_emissions_for_row(row, default_weight_g, washes_per_month)
        emissions_results.append(result)
    
    # Add emissions columns to dataframe
    df['total_kgco2e'] = [r['total_kgco2e'] for r in emissions_results]
    df['materials_kgco2e'] = [r['materials_kgco2e'] for r in emissions_results]
    df['manufacturing_kgco2e'] = [r['manufacturing_kgco2e'] for r in emissions_results]
    df['washing_kgco2e'] = [r['washing_kgco2e'] for r in emissions_results]
    df['parsed_country'] = [r['parsed_country'] for r in emissions_results]
    df['parsed_materials'] = [json.dumps(r['parsed_materials']) for r in emissions_results]
    df['calculation_error'] = [r['calculation_error'] for r in emissions_results]
    
    # Save results
    df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    successful = df['total_kgco2e'].notna().sum()
    failed = df['total_kgco2e'].isna().sum()
    
    print(f"\nSuccessful calculations: {successful}/{len(df)} ({successful/len(df)*100:.1f}%)")
    print(f"Failed calculations: {failed}/{len(df)} ({failed/len(df)*100:.1f}%)")
    
    if successful > 0:
        print(f"\nEmissions Statistics (kg CO2e):")
        print(f"  Total Emissions:")
        print(f"    Mean:   {df['total_kgco2e'].mean():.3f}")
        print(f"    Median: {df['total_kgco2e'].median():.3f}")
        print(f"    Min:    {df['total_kgco2e'].min():.3f}")
        print(f"    Max:    {df['total_kgco2e'].max():.3f}")
        
        print(f"\n  Materials Emissions:")
        print(f"    Mean:   {df['materials_kgco2e'].mean():.3f}")
        
        print(f"\n  Manufacturing Emissions:")
        print(f"    Mean:   {df['manufacturing_kgco2e'].mean():.3f}")
        
        print(f"\n  Washing Emissions (lifetime):")
        print(f"    Mean:   {df['washing_kgco2e'].mean():.3f}")
    
    # Show breakdown by component
    if successful > 0:
        print(f"\n{'='*80}")
        print("EMISSIONS BREAKDOWN (% of total)")
        print(f"{'='*80}")
        
        total_materials = df['materials_kgco2e'].sum()
        total_manufacturing = df['manufacturing_kgco2e'].sum()
        total_washing = df['washing_kgco2e'].sum()
        grand_total = total_materials + total_manufacturing + total_washing
        
        print(f"\nMaterials:      {total_materials/grand_total*100:.1f}%")
        print(f"Manufacturing:  {total_manufacturing/grand_total*100:.1f}%")
        print(f"Washing:        {total_washing/grand_total*100:.1f}%")
    
    # Show common errors
    if failed > 0:
        print(f"\n{'='*80}")
        print("COMMON ERRORS")
        print(f"{'='*80}")
        errors = df[df['calculation_error'].notna()]['calculation_error'].value_counts()
        for error, count in errors.head(5).items():
            print(f"  {error}: {count} occurrences")
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_csv}")
    print(f"{'='*80}\n")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate CO2 emissions from OCR evaluation CSV")
    parser.add_argument("--input", type=str, default="ocr_evaluation_easyocr_dec1.csv",
                       help="Input CSV file with OCR predictions")
    parser.add_argument("--output", type=str, default="emissions_estimates_from_ocr.csv",
                       help="Output CSV file with emissions estimates")
    parser.add_argument("--weight_g", type=float, default=1000.0,
                       help="Default garment weight in grams (default: 1000g = 1kg)")
    parser.add_argument("--washes_per_month", type=float, default=2.0,
                       help="Number of washes per month (default: 2.0)")
    
    args = parser.parse_args()
    
    df = process_ocr_evaluation_csv(
        input_csv=args.input,
        output_csv=args.output,
        default_weight_g=args.weight_g,
        washes_per_month=args.washes_per_month
    )
