import argparse
import csv
import json
import os
import sys
import time

import imagehash
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR      = os.path.join(ROOT, "data")
DATASET_CSV   = os.path.join(DATA_DIR, "dataset.csv")
CLEAN_CSV     = os.path.join(DATA_DIR, "inputs", "dataset_clean.csv")
FEATURES_DIR  = os.path.join(DATA_DIR, "features")
METADATA_JSON = os.path.join(DATA_DIR, "metadata.json")


def compute_phash(image_path: str, hash_size: int = 8) -> str:
    try:
        return str(imagehash.phash(Image.open(image_path), hash_size=hash_size))
    except Exception:
        return ""


def _resolve_csv():
    if os.path.exists(CLEAN_CSV):
        return CLEAN_CSV
    if os.path.exists(DATASET_CSV):
        return DATASET_CSV
    raise FileNotFoundError("Dataset CSV tidak ditemukan. Jalankan: php artisan cbir:sync")


def _extract_all_subfeatures(image_path: str, extractor, meth: str) -> dict:
    """
    Ekstrak SEMUA sub-fitur dari satu gambar secara lengkap.

    - Jika extractor adalah UltraCombinedExtractor  → gunakan semua komponen internal.
    - Jika extractor adalah CombinedExtractor        → gunakan _deep/_color/_texture.
    - Jika extractor individual                      → ekstrak dan tulis ke slot yang sesuai.

    Return dict berisi semua key fitur yang tersimpan di JSON.
    """
    import numpy as np
    from src.addons.extraction.extractor import UltraCombinedExtractor, CombinedExtractor

    empty = np.array([], dtype=np.float32)

    # ── ULTRA: semua komponen tersedia ──────────────────────────────────────
    if isinstance(extractor, UltraCombinedExtractor):
        resnet_f   = extractor._resnet.extract(image_path)
        effnet_f   = extractor._effnet.extract(image_path)
        vgg_f      = extractor._vgg.extract(image_path)
        hog_f      = extractor._hog.extract(image_path)
        color_hsv  = extractor._color_hsv.extract(image_path)
        color_rgb  = extractor._color_rgb.extract(image_path)
        gabor_f    = extractor._gabor.extract(image_path)
        lbp_f      = extractor._lbp.extract(image_path)
        sift_f     = extractor._sift.extract(image_path)
        akaze_f    = extractor._akaze.extract(image_path)
        dominant_f = extractor._dominant.extract(image_path)

        combined_vec = np.concatenate([
            resnet_f   * 0.40,
            effnet_f   * 0.20,
            vgg_f      * 0.10,
            hog_f      * 0.10,
            color_hsv  * 0.10,
            color_rgb  * 0.05,
            gabor_f    * 0.05,
            lbp_f      * 0.05,
            sift_f     * 0.025,
            akaze_f    * 0.015,
            dominant_f * 0.010,
        ]).astype(np.float32)

        return {
            # Vektor utama — dipakai oleh app.py / CBIR engine
            "combined"          : combined_vec.tolist(),
            "combined_features" : combined_vec.tolist(),
            # Deep Learning
            "deep"              : resnet_f.tolist(),
            "deep_features"     : resnet_f.tolist(),
            "resnet50"          : resnet_f.tolist(),
            "efficientnet"      : effnet_f.tolist(),
            "vgg16"             : vgg_f.tolist(),
            # Color
            "color"             : color_hsv.tolist(),
            "color_histogram"   : color_hsv.tolist(),
            "rgb_histogram"     : color_rgb.tolist(),
            "dominant_color"    : dominant_f.tolist(),
            # Texture & Shape
            "texture_features"  : lbp_f.tolist(),
            "lbp"               : lbp_f.tolist(),
            "gabor"             : gabor_f.tolist(),
            "hog"               : hog_f.tolist(),
            # Keypoint
            "sift"              : sift_f.tolist(),
            "akaze"             : akaze_f.tolist(),
            # Meta
            "method"            : meth,
            "dim_combined"      : int(combined_vec.shape[0]),
            "dim_deep"          : int(resnet_f.shape[0]),
            "dim_color"         : int(color_hsv.shape[0]),
        }

    # ── COMBINED: ResNet50 + ColorHist + LBP ────────────────────────────────
    if isinstance(extractor, CombinedExtractor):
        deep_feat    = extractor._deep.extract(image_path)
        color_feat   = extractor._color.extract(image_path)
        texture_feat = extractor._texture.extract(image_path)
        combined_vec = np.concatenate([
            deep_feat    * 0.70,
            color_feat   * 0.20,
            texture_feat * 0.10,
        ]).astype(np.float32)

        return {
            "combined"          : combined_vec.tolist(),
            "combined_features" : combined_vec.tolist(),
            "deep"              : deep_feat.tolist(),
            "deep_features"     : deep_feat.tolist(),
            "resnet50"          : deep_feat.tolist(),
            "color"             : color_feat.tolist(),
            "color_histogram"   : color_feat.tolist(),
            "texture_features"  : texture_feat.tolist(),
            "lbp"               : texture_feat.tolist(),
            # placeholder kosong untuk key yang tidak diekstrak
            "efficientnet"      : [],
            "rgb_histogram"     : [],
            "dominant_color"    : [],
            "gabor"             : [],
            "hog"               : [],
            "sift"              : [],
            "akaze"             : [],
            "method"            : meth,
            "dim_combined"      : int(combined_vec.shape[0]),
            "dim_deep"          : int(deep_feat.shape[0]),
            "dim_color"         : int(color_feat.shape[0]),
        }

    # ── INDIVIDUAL METHOD ────────────────────────────────────────────────────
    vec = extractor.extract(image_path)

    deep_f    = vec if meth in ("deep", "resnet50")                  else empty
    effnet_f  = vec if meth == "efficientnet"                        else empty
    vgg_f     = vec if meth == "vgg16"                               else empty
    color_f   = vec if meth in ("color", "color_histogram")          else empty
    rgb_f     = vec if meth == "rgb_histogram"                       else empty
    dom_f     = vec if meth == "dominant_color"                      else empty
    lbp_f     = vec if meth == "lbp"                                 else empty
    gabor_f   = vec if meth == "gabor"                               else empty
    hog_f     = vec if meth == "hog"                                 else empty
    sift_f    = vec if meth == "sift"                                else empty
    akaze_f   = vec if meth in ("akaze", "orb")                      else empty

    return {
        "combined"          : vec.tolist(),
        "combined_features" : vec.tolist(),
        "deep"              : deep_f.tolist(),
        "deep_features"     : deep_f.tolist(),
        "resnet50"          : deep_f.tolist(),
        "efficientnet"      : effnet_f.tolist(),
        "vgg16"             : vgg_f.tolist(),
        "color"             : color_f.tolist(),
        "color_histogram"   : color_f.tolist(),
        "rgb_histogram"     : rgb_f.tolist(),
        "dominant_color"    : dom_f.tolist(),
        "texture_features"  : lbp_f.tolist(),
        "lbp"               : lbp_f.tolist(),
        "gabor"             : gabor_f.tolist(),
        "hog"               : hog_f.tolist(),
        "sift"              : sift_f.tolist(),
        "akaze"             : akaze_f.tolist(),
        "method"            : meth,
        "dim_combined"      : int(vec.shape[0]),
        "dim_deep"          : int(deep_f.shape[0]),
        "dim_color"         : int(color_f.shape[0]),
    }


def build_features(method="combined", csv_path=None, app_url="http://127.0.0.1:8000"):
    """
    Ekstrak fitur dari semua gambar (products + packages) di dataset CSV.
    Hasil disimpan ke data/features/features_<method>.json
    dan data/metadata.json (dipakai Flask API).

    Args:
        method  : Nama extractor. Gunakan 'all' untuk semua metode.
        csv_path: Path ke CSV. None = cari otomatis.
        app_url : Laravel APP_URL untuk image_url.
    """
    from src.addons.extraction.extractor import extractors, get_extractor
    from src.addons.data import load_dataset_csv, save_feature_database, resolve_portable_path

    csv_file = csv_path or _resolve_csv()
    rows     = load_dataset_csv(csv_file)

    products = [r for r in rows if r.get("Type", "").strip().lower() == "product"]
    packages = [r for r in rows if r.get("Type", "").strip().lower() == "package"]

    methods = list(extractors.keys()) if method == "all" else [method]

    print("=" * 60)
    print("BUILD FEATURES")
    print("=" * 60)
    print("CSV      : " + csv_file)
    print("Metode   : " + str(methods))
    print("Products : " + str(len(products)))
    print("Packages : " + str(len(packages)))
    print("Total    : " + str(len(rows)))
    print()

    os.makedirs(FEATURES_DIR, exist_ok=True)
    results = {}

    for meth in methods:
        print("[" + meth.upper() + "] Memuat extractor...")
        extractor = get_extractor(meth)
        db        = {"images": []}
        total     = 0
        skipped   = 0
        t_start   = time.perf_counter()

        for row in rows:
            owner_id    = row.get("ID", "").strip()
            item_type   = row.get("Type", "product").strip().lower()
            name        = row.get("Name", "").strip()
            category    = row.get("Category", "unknown").strip().lower()
            price       = row.get("Price", "0").strip()
            disc_price  = row.get("Discount_Price", "").strip()
            image_path  = row.get("Image_Path", "").strip()
            description = row.get("Description", "").strip()

            if item_type not in ("product", "package"):
                item_type = "product"

            image_path = image_path.replace("/", os.sep).replace("\\", os.sep)
            image_path = resolve_portable_path(image_path)

            if not image_path or not os.path.exists(image_path):
                print("  SKIP (not found): " + image_path)
                skipped += 1
                continue

            if os.path.getsize(image_path) > 10 * 1024 * 1024:
                skipped += 1
                continue

            # Build image_url
            try:
                norm  = image_path.replace("\\", "/")
                parts = norm.split("/storage/app/public/")
                if len(parts) == 2:
                    image_url = app_url + "/storage/" + parts[1]
                else:
                    segs = norm.split("/")
                    image_url = app_url + "/storage/" + segs[-2] + "/" + segs[-1]
            except Exception:
                image_url = ""

            try:
                feat_dict = _extract_all_subfeatures(image_path, extractor, meth)

                entry = {
                    "id"      : total + 1,
                    "path"    : image_path,
                    "phash"   : compute_phash(image_path),
                    "metadata": {
                        "type"          : item_type,
                        "owner_id"      : int(owner_id) if owner_id.isdigit() else None,
                        "name"          : name,
                        "category"      : category,
                        "price"         : float(price) if price else 0.0,
                        "discount_price": float(disc_price) if disc_price else 0.0,
                        "description"   : description,
                        "image_url"     : image_url,
                        "image_path"    : image_path,
                    },
                    "features": feat_dict,
                }

                db["images"].append(entry)
                total += 1
                print(
                    "  [" + item_type.upper().ljust(8) + "] "
                    "ID=" + owner_id.ljust(4) + " | "
                    + category.ljust(15) + " | " + name
                )

            except Exception as e:
                print("  ERROR: ID=" + owner_id + " " + name + ": " + str(e))
                skipped += 1

        elapsed = round(time.perf_counter() - t_start, 2)

        # Simpan ke features/features_<meth>.json
        feat_path = os.path.join(FEATURES_DIR, "features_" + meth + ".json")
        save_feature_database(db, feat_path)

        # Update metadata.json jika combined atau ultra (dipakai Flask API)
        if meth in ("combined", "ultra"):
            save_feature_database(db, METADATA_JSON)
            print("  metadata.json diperbarui: " + METADATA_JSON)

        results[meth] = {"total": total, "skipped": skipped, "elapsed": elapsed, "path": feat_path}
        print(
            "  [" + meth + "] "
            + str(total) + " item, "
            + str(skipped) + " diskip ("
            + str(elapsed) + "s)"
        )

    # Ringkasan akhir
    if db.get("images"):
        types = {}
        cats  = {}
        for img in db["images"]:
            t = img["metadata"]["type"]
            c = img["metadata"]["category"]
            types[t] = types.get(t, 0) + 1
            cats[c]  = cats.get(c, 0) + 1
        print()
        print("=" * 60)
        print("BUILD FEATURES SELESAI")
        print("  Tipe     : " + str(types))
        print("  Kategori : " + str(cats))
        print("=" * 60)

    return results


if __name__ == "__main__":
    from src.addons.extraction.extractor import extractors
    parser = argparse.ArgumentParser(description="Ekstrak fitur CBIR dari dataset CSV Laravel")
    parser.add_argument(
        "--method", default="all",
        choices=list(extractors.keys()) + ["all"],
        help="Metode ekstraksi. Default='all' (semua method sekaligus)."
    )
    parser.add_argument("--csv",     default=None, help="Path ke dataset CSV")
    parser.add_argument("--app-url", default="http://127.0.0.1:8000", help="Laravel APP_URL")
    args = parser.parse_args()
    build_features(method=args.method, csv_path=args.csv, app_url=args.app_url)