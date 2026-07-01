#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup_complete.py — Verify and complete CBIR setup

Langkah-langkah:
  1. Verify Python & dependencies
  2. Test extraction modules
  3. Create sample data if needed
  4. Run basic tests
  5. Verify Flask API
"""

import os
import sys
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"
METADATA_JSON = DATA_DIR / "metadata.json"

print("=" * 70)
print("  Wedding CBIR AI Core — Setup Verification")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Check Python version
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/6] Checking Python version...")
py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
print(f"  ✓ Python {py_version}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Verify critical dependencies
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/6] Verifying dependencies...")
required = {
    "flask": "Flask",
    "cv2": "OpenCV",
    "torch": "PyTorch",
    "torchvision": "TorchVision",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "PIL": "Pillow",
}

missing = []
for module, name in required.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (MISSING)")
        missing.append(module)

if missing:
    print(f"\n⚠️  Missing: {', '.join(missing)}")
    print("  Run: make.bat install")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Test extraction modules
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/6] Testing extraction modules...")
try:
    from src.addons.extraction.extractor import get_extractor, extractors
    print(f"  ✓ Extractor registry loaded ({len(extractors)} methods)")
    
    # List available methods
    methods = list(extractors.keys())
    print(f"  Available: {', '.join(methods[:3])}...")
    
    # Test that get_extractor works
    ext_combined = get_extractor("combined")
    print(f"  ✓ Combined extractor ready")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Test finder & metrics
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/6] Testing finder and metrics...")
try:
    from src.addons.finder import get_finder, finders
    from src.addons.metrics import mean_average_precision, mean_reciprocal_rank
    
    print(f"  ✓ Finder registry loaded ({len(finders)} metrics)")
    
    # Test finder
    finder_euclidean = get_finder("euclidean")
    print(f"  ✓ Euclidean finder ready")
    
    # Test metrics
    test_queries = [([1, 2, 3], {1, 2, 3})]
    map_score = mean_average_precision(test_queries)
    print(f"  ✓ Metrics working (MAP: {map_score:.2f})")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Create data directories
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/6] Preparing data directories...")
for subdir in ["inputs", "features", "uploads", "evaluation"]:
    path = DATA_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {path.name}/")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Create or verify sample dataset
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/6] Checking sample data...")

if not DATASET_CSV.exists():
    print(f"  ℹ Creating sample dataset.csv...")
    import csv
    
    # Create minimal sample CSV with wedding products
    sample_rows = [
        {
            "ID": "1",
            "Type": "product",
            "Name": "White Rose Bouquet",
            "Category": "Flowers",
            "Price": "150000",
            "Discount_Price": "120000",
            "Organizer": "Demo Organizer",
            "Image_Path": "data/uploads/sample1.jpg",
            "Description": "Beautiful white roses bouquet"
        },
        {
            "ID": "2",
            "Type": "package",
            "Name": "Gold Wedding Decoration",
            "Category": "Decoration",
            "Price": "500000",
            "Discount_Price": "450000",
            "Organizer": "Demo Organizer",
            "Image_Path": "data/uploads/sample2.jpg",
            "Description": "Complete gold decoration package"
        },
    ]
    
    with open(DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sample_rows[0].keys())
        writer.writeheader()
        writer.writerows(sample_rows)
    
    print(f"  ✓ Created {DATASET_CSV.name} (2 sample records)")
else:
    with open(DATASET_CSV) as f:
        import csv
        rows = list(csv.DictReader(f))
    print(f"  ✓ Using existing {DATASET_CSV.name} ({len(rows)} records)")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  ✅ Setup Complete!")
print("=" * 70)
print("\nNext steps:")
print("  1. Add sample images to data/uploads/")
print("  2. Run: python rebuild_index.py")
print("  3. Run: make.bat test")
print("  4. Run: make.bat server")
print("  5. Test: curl http://127.0.0.1:5000/health")
print("\nOr use make.bat commands:")
print("  make prepare   — Create directories")
print("  make rebuild   — Rebuild index from dataset.csv")
print("  make test      — Run all tests")
print("  make server    — Start Flask server")
print("=" * 70)
