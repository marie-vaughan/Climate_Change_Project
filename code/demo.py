import argparse, json, sys
from parser import parse_from_text
from calc import estimate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--image", type=str, action="append",
                    help="Path to tag image(s); use multiple --image for multi-photo tags")
    ap.add_argument("--type", type=str, default="tshirt")
    ap.add_argument("--weight_g", type=float, default=None)
    ap.add_argument("--wash_per_month", type=float, default=2.0)
    ap.add_argument("--mode", type=str, default="ship")
    ap.add_argument("--lang", type=str, default="eng")
    ap.add_argument("--show_ocr", action="store_true",
                    help="Print merged OCR text before parsing")
    args = ap.parse_args()

    if args.image:
        try:
            from ocr import run_ocr_many
            # Run PaddleOCR on all tag images
            args.text = run_ocr_many(args.image)

            if args.show_ocr:
                print("\n=== RAW OCR TEXT ===")
                print(args.text)
                print("====================\n")

        except Exception as e:
            print(f"❌ OCR failed: {e}")
            sys.exit(1)

    if not args.text:
        sys.exit("Provide --text or at least one --image")

    record = parse_from_text(
        args.text,
        garment_type=args.type,
        default_weight_g=args.weight_g,
        washes_per_month=args.wash_per_month
    )
    res = estimate(record, preferred_mode=args.mode)

    print("=== PARSED TAG ===")
    print(json.dumps({
        "materials": [{"fiber": m.fiber, "pct": m.pct} for m in record.materials],
        "origin": record.origin_country,
        "care": vars(record.care)
    }, indent=2))

    print("\n=== RESULTS ===")
    print(json.dumps({
        "total": round(res.total_kgco2e, 3),
        "breakdown": {k: round(v, 3) for k, v in res.breakdown.items()},
        "assumptions": res.assumptions
    }, indent=2))

if __name__ == "__main__":
    main()

