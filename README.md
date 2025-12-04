# EcoTag: Estimating Garment Carbon Footprint from Clothing Labels

## Overview

This project estimates the carbon footprint of clothing items by analyzing garment care tags using Optical Character Recognition (OCR). The OCR model extracts information about materials, manufacturing origin, and care instructions from clothing labels to calculate total CO₂ emissions across a garment's lifecycle.

## To Run Code

### Prerequisites

- Python 3.8+
- pip

### Dependencies

Install required packages:

```bash
pip install easyocr opencv-python numpy pandas matplotlib seaborn tqdm codecarbon pillow pytesseract paddleocr
```

### Usage

To see the CO2 emissions of a single tag image:

```bash
cd code
python demo.py --image ../cropped_tags/[IMG_NAME].JPG
```

Example output:
```json
=== PARSED TAG ===
{
  "materials": [
    {"fiber": "cotton", "pct": 100.0}
  ],
  "origin": "guatemala",
  "care": {
    "wash": "cold",
    "dry": "tumble",
    "dry_clean": "none",
    "washes_per_month": 2.0
  }
}

=== RESULTS ===
{
  "total": 12.456,
  "breakdown": {
    "materials": 5.230,
    "manufacturing": 0.500,
    "washing": 6.726
  },
  "assumptions": {
    "weight_g": "1000",
    "origin": "guatemala",
    "washes_lifetime": "48"
  }
}
```


To see OCR text reading results for all the tag images and evaluate accuracy:

EasyOCR:
```bash
python code/evaluate_easyocr_Dec1.py
```

Pytesseract:
```bash
python code/evaluate_pytesseract_Dec1.py
```

PaddleOCR:
```bash
python code/evaluate_paddleocr_Dec1.py
```

Outputs for each of the OCR models are saved in CSV files that show accuracy metrics for material detection, country extraction, and care instruction parsing.





## Additional Tools


### Generate Synthetic Tags

Create test data:
```bash
python code/generate_synthetic_tagsnovtwentyfifth.py
```


## CO₂ Emission Factors

The calculation is based on emission factors in `co2factors.py`:
- **Materials**: kg CO₂/kg for cotton, polyester, wool, etc.
- **Manufacturing**: kg CO₂/kg by country
- **Washing**: kg CO₂ per wash cycle (cold, warm, hot)
- **Drying**: kg CO₂ per tumble dry cycle

Default assumptions:
- Garment lifetime: 2 years
- Washes per month: 2
- Total washes: 48

CO₂ emission factors based on apparel lifecycle assessment literature

## Carbon Tracking

This project uses CodeCarbon to track computational emissions during OCR processing. Logs are saved to `codecarbon_logs/emissions.csv`.
