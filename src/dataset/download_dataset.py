import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DATA_DIR    = os.path.join(ROOT, "data")
DATASET_CSV = os.path.join(DATA_DIR, "dataset.csv")
DEFAULT_STORAGE = os.path.join(os.path.dirname(ROOT), "Admin-Panel-Mobile", "storage", "app", "public")

def prepare_dataset(laravel_storage=DEFAULT_STORAGE):
    print("=" * 60)
    print("PREPARE DATASET - Verifikasi dari Laravel Storage")
    print("Storage : " + laravel_storage)
    print("CSV     : " + DATASET_CSV)
    print("=" * 60)
    if not os.path.exists(DATASET_CSV):
        print("ERROR: dataset.csv tidak ditemukan. Jalankan: php artisan cbir:sync")
        sys.exit(1)
    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total    = len(rows)
    found    = 0
    missing  = 0
    products = 0
    packages = 0
    for row in rows:
        img = row.get("Image_Path","").strip().replace("/",os.sep).replace("\\",os.sep)
        t   = row.get("Type","").strip().lower()
        if os.path.exists(img):
            found += 1
            if t == "product":
                products += 1
            elif t == "package":
                packages += 1
            print("  OK   [" + t.upper().ljust(8) + "] " + row.get("Name", ""))
        else:
            missing += 1
            print("  MISS [" + t.upper().ljust(8) + "] " + row.get("Name", "") + " -> " + img)
    print()
    print("Total="+str(total)+" Found="+str(found)+" Missing="+str(missing))
    print("Products="+str(products)+" Packages="+str(packages))
    return {"total":total,"found":found,"missing":missing,"products":products,"packages":packages}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--laravel-path", default=DEFAULT_STORAGE)
    args = parser.parse_args()
    prepare_dataset(laravel_storage=args.laravel_path)
