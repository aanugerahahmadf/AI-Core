#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_sample_images.py — Create minimal test images for CBIR

Generates simple placeholder images untuk testing tanpa perlu dataset real.
"""

import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_sample_image(name: str, color: tuple, text: str) -> str:
    """Create a simple colored image with text for testing."""
    img = Image.new("RGB", (256, 256), color=color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw text in center
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (256 - text_width) // 2
    y = (256 - text_height) // 2
    
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    # Save
    path = DATA_DIR / name
    img.save(path)
    return str(path)

print("Creating sample images for CBIR testing...")

# Create 4 sample images with different colors (representing wedding themes)
samples = [
    ("sample1_white.jpg", (220, 220, 220), "White Flowers"),
    ("sample2_gold.jpg", (218, 165, 32), "Gold Deco"),
    ("sample3_pink.jpg", (255, 192, 203), "Pink Theme"),
    ("sample4_blue.jpg", (173, 216, 230), "Blue Theme"),
]

for filename, color, text in samples:
    path = create_sample_image(filename, color, text)
    print(f"  ✓ Created {path}")

print("\n✅ Sample images created in data/uploads/")
print("   Next: Run 'python rebuild_index.py' to extract features")
