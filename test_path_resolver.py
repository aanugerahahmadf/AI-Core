"""
test_path_resolver.py - Tes manual resolve_portable_path ke Mobile-App
Jalankan: python test_path_resolver.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.addons.data import resolve_portable_path

MOBILE_APP = r"D:\Weeding-Organizer-CBIR\Mobile-App"

print("=" * 60)
print("  TEST: resolve_portable_path -> Mobile-App")
print("=" * 60)
print("  Mobile-App exists : " + str(os.path.exists(MOBILE_APP)))
print()

test_cases = [
    # Path lama dari Admin-Panel-Mobile
    r"D:\Weeding-Organizer-CBIR\Admin-Panel-Mobile\storage\app\public\4\product-1.png",
    # Path dari Mobile-App langsung (harus langsung ketemu)
    r"D:\Weeding-Organizer-CBIR\Mobile-App\storage\app\public\4\product-1.png",
    # Path kosong
    "",
]

for i, path in enumerate(test_cases, 1):
    resolved = resolve_portable_path(path)
    exists   = os.path.exists(resolved) if resolved else False
    if exists:
        status = "FOUND"
    elif resolved:
        status = "not found (file belum ada)"
    else:
        status = "empty"

    print("[TEST " + str(i) + "]")
    print("  Input   : " + (path or "(empty)"))
    print("  Resolved: " + (resolved or "(empty)"))
    print("  Status  : " + status)
    print()
