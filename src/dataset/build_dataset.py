import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DATA_DIR    = os.path.join(ROOT, "data")
DATASET_CSV = os.path.join(DATA_DIR, "dataset.csv")
CLEAN_CSV   = os.path.join(DATA_DIR, "inputs", "dataset_clean.csv")

def build_dataset(csv_path=None, output_path=None, max_size_mb=10.0):
    from src.addons.data import load_dataset_csv, dataset_stats
    csv_path    = csv_path    or DATASET_CSV
    output_path = output_path or CLEAN_CSV
    print("=" * 60)
    print("BUILD DATASET - Validasi dari dataset.csv Laravel")
    print("Input  : " + csv_path)
    print("Output : " + output_path)
    print("=" * 60)
    rows = load_dataset_csv(csv_path)
    stats_in = dataset_stats(rows)
    print("Total  : " + str(len(rows)))
    print("Tipe   : " + str(stats_in["per_type"]))
    max_bytes = max_size_mb * 1024 * 1024
    valid   = []
    skipped = 0
    for row in rows:
        img = row.get("Image_Path", "").strip().replace("/", os.sep).replace("\\", os.sep)
        if not img or not os.path.exists(img):
            skipped += 1
            continue
        if os.path.getsize(img) > max_bytes:
            skipped += 1
            continue
        row["Image_Path"] = img
        valid.append(row)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        if valid:
            writer = csv.DictWriter(f, fieldnames=valid[0].keys())
            writer.writeheader()
            writer.writerows(valid)
    stats_out = dataset_stats(valid)
    print("Valid  : " + str(len(valid)) + " | Diskip: " + str(skipped))
    print("Tipe   : " + str(stats_out["per_type"]))
    print("Saved  : " + output_path)
    return {"total_input": len(rows), "total_valid": len(valid), "skipped": skipped, "stats": stats_out}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DATASET_CSV)
    parser.add_argument("--output", default=CLEAN_CSV)
    parser.add_argument("--max-size-mb", type=float, default=10.0)
    args = parser.parse_args()
    build_dataset(csv_path=args.csv, output_path=args.output, max_size_mb=args.max_size_mb)
