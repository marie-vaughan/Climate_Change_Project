# summarize_emissions.py
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def summarize_emissions():
    """
    Aggregate all emissions reports and create summary
    """
    
    emissions_dir = Path('emissions')
    
    if not emissions_dir.exists():
        print("No emissions logs found!")
        return
    
    # Load all JSON reports
    reports = []
    for json_file in emissions_dir.glob('*_report.json'):
        with open(json_file, 'r') as f:
            reports.append(json.load(f))
    
    if not reports:
        print("No emission reports found!")
        return
    
    # Calculate totals
    total_emissions_kg = sum(r['emissions_kg_co2'] for r in reports)
    total_energy_kwh = sum(r['energy_consumed_kwh'] for r in reports)
    total_duration = sum(r['duration_seconds'] for r in reports)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE PIPELINE EMISSIONS SUMMARY")
    print(f"{'='*60}\n")
    
    # Per-step breakdown
    print("Step-by-step breakdown:")
    print(f"{'='*60}")
    
    for report in sorted(reports, key=lambda x: x['timestamp']):
        step = report['step']
        emissions_g = report['emissions_kg_co2'] * 1000
        duration_min = report['duration_seconds'] / 60
        
        print(f"\n{step.upper().replace('_', ' ')}:")
        print(f"  Duration: {duration_min:.1f} minutes")
        print(f"  CO2: {emissions_g:.2f} g")
        print(f"  Energy: {report['energy_consumed_kwh']:.4f} kWh")
        
        if 'num_images' in report:
            print(f"  Images: {report['num_images']}")
        elif 'output_images' in report:
            print(f"  Images: {report['output_images']}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL PIPELINE EMISSIONS")
    print(f"{'='*60}")
    print(f"Total Duration: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
    print(f"Total CO2 Emissions: {total_emissions_kg*1000:.2f} g ({total_emissions_kg:.6f} kg)")
    print(f"Total Energy: {total_energy_kwh:.4f} kWh")
    
    print(f"\n{'='*60}")
    print(f"ENVIRONMENTAL EQUIVALENTS")
    print(f"{'='*60}")
    print(f"🚗 Miles driven: {total_emissions_kg*2.5:.2f} miles")
    print(f"🌳 Trees needed (1 year): {total_emissions_kg*0.06:.2f} trees")
    print(f"📱 Smartphone charges: {total_emissions_kg*120:.0f} charges")
    print(f"💡 LED bulb hours: {total_energy_kwh/0.01:.0f} hours")
    print(f"🏠 Home electricity (avg day): {total_energy_kwh/30:.2f} days")
    
    # Save combined summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_emissions_kg_co2': total_emissions_kg,
        'total_energy_kwh': total_energy_kwh,
        'total_duration_seconds': total_duration,
        'steps': reports,
        'equivalents': {
            'miles_driven': total_emissions_kg * 2.5,
            'trees_needed_yearly': total_emissions_kg * 0.06,
            'smartphone_charges': total_emissions_kg * 120,
            'led_bulb_hours': total_energy_kwh / 0.01,
        }
    }
    
    with open('emissions/complete_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Complete summary saved: emissions/complete_summary.json")
    print(f"{'='*60}\n")
    
    # Create CSV for easy analysis
    df = pd.DataFrame(reports)
    df.to_csv('emissions/emissions_breakdown.csv', index=False)
    print(f"✓ CSV saved: emissions/emissions_breakdown.csv\n")

if __name__ == "__main__":
    summarize_emissions()