# -*- coding: utf-8 -*-
"""
app.py — Wedding CBIR Flask API Server v2.0

Terhubung ke:
  - src/addons/         : feature extraction & similarity search
  - ai_sync.py          : sync fitur ke dataset.csv
  - Laravel Mobile-App via HTTP (CBIRService.php)

Endpoints yang dipanggil Laravel (CBIRService.php & CBIRController.php):
  POST /api/search                    ← CBIRService::searchByImage()
  POST /api/index/add                 ← CBIRService::indexMedia()
  POST /api/index/remove              ← CBIRService::removeFromIndex()
  POST /api/index/rebuild-from-dataset← SyncCbirCsv.php / SyncAICoreCommand.php
  POST /api/index/clear               ← admin panel
  GET  /status                        ← CBIRController::getStats()
  GET  /health                        ← health check
  GET  /api/index/stats               ← CBIRController::getStats()
  POST /api/features/extract          ← debugging
  POST /api/sync                      ← ai_sync.py trigger (php artisan ai:sync)
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import warnings

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

import imagehash
import werkzeug.serving
original_log = werkzeug.serving._log
werkzeug.serving._log = lambda type, msg, *a, **kw: None if 'development server' in str(msg) else original_log(type, msg, *a, **kw)
warnings.filterwarnings('ignore', message='.*development server.*', module='werkzeug')

# ---------------------------------------------------------------------------
# Path setup & .env
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from src.addons.data import load_feature_database, save_feature_database, resolve_portable_path
from src.addons.extraction.extractor import get_extractor
from src.addons.finder import get_finder
from src.addons.image_arithmetic import arithmetic_search, OPS
from src.addons.rerank import rerank_by_ensemble
import numpy as np

# ---------------------------------------------------------------------------
# Legacy CBIREngine Class (Backward Compatibility)
# ---------------------------------------------------------------------------

class CBIREngine:
    def __init__(self, database_path="data"):
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.database_path = database_path
        self.metadata_path = os.path.join(database_path, "metadata.json")
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.feature_extractor.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.load_database()

    def load_database(self):
        from src.addons.data import load_feature_database
        self.database = load_feature_database(self.metadata_path)

    def save_database(self):
        from src.addons.data import save_feature_database
        save_feature_database(self.database, self.metadata_path)

    def extract_deep_features(self, image_path):
        import torch
        try:
            img = Image.open(image_path).convert("RGB")
            t = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = self.feature_extractor(t)
            return feat.squeeze().numpy().astype(np.float32)
        except Exception as e:
            print("[WARN] deep: " + str(e))
            return np.zeros(2048, dtype=np.float32)

    def extract_color_histogram(self, image_path):
        import cv2
        try:
            img = cv2.imread(image_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten().astype(np.float32)
        except Exception as e:
            print("[WARN] color: " + str(e))
            return np.zeros(512, dtype=np.float32)

    def extract_texture_features(self, image_path):
        import cv2
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            lbp = np.zeros_like(img, dtype=np.uint8)
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    c    = int(img[i, j])
                    code = 0
                    code |= (int(img[i-1, j-1]) > c) << 7
                    code |= (int(img[i-1, j])   > c) << 6
                    code |= (int(img[i-1, j+1]) > c) << 5
                    code |= (int(img[i,   j+1]) > c) << 4
                    code |= (int(img[i+1, j+1]) > c) << 3
                    code |= (int(img[i+1, j])   > c) << 2
                    code |= (int(img[i+1, j-1]) > c) << 1
                    code |= (int(img[i,   j-1]) > c) << 0
                    lbp[i, j] = code
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-7
            return hist
        except Exception as e:
            print("[WARN] texture: " + str(e))
            return np.zeros(256, dtype=np.float32)

    def extract_all_features(self, image_path):
        deep    = self.extract_deep_features(image_path)
        color   = self.extract_color_histogram(image_path)
        texture = self.extract_texture_features(image_path)
        combined = np.concatenate([deep*0.70, color*0.20, texture*0.10])
        return {
            "deep_features": deep.tolist(), "color_histogram": color.tolist(),
            "texture_features": texture.tolist(), "combined_features": combined.tolist(),
        }

    def calculate_euclidean_distance(self, a, b):
        from scipy.spatial import distance
        return float(distance.euclidean(a, b))

    def calculate_similarity_score(self, dist, max_dist=25.0):
        linear = max(0.0, 100.0-(dist/max_dist*100.0))
        sim = (linear/100.0)**2*100.0
        return round(sim if sim >= 15.0 else 0.0, 2)

    def add_image_to_database(self, image_path, metadata):
        # Validasi type — hanya product atau package
        if metadata.get("type") not in ("product", "package"):
            metadata["type"] = "product"
        features = self.extract_all_features(image_path)
        entry = {
            "id": len(self.database["images"])+1,
            "path": image_path, "metadata": metadata, "features": features,
        }
        self.database["images"].append(entry)
        self.save_database()
        return entry

    def search_similar_images(self, query_path, top_k=10):
        query_feat = np.array(
            self.extract_all_features(query_path)["combined_features"], dtype=np.float32
        )
        results = []
        for img in self.database["images"]:
            db_feat = np.array(img["features"]["combined_features"], dtype=np.float32)
            dist = self.calculate_euclidean_distance(query_feat, db_feat)
            sim  = self.calculate_similarity_score(dist)
            results.append({
                "id": img["id"], "path": img["path"],
                "metadata": img["metadata"], "distance": dist, "similarity": sim,
            })
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR      = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
FEATURE_DB    = os.path.join(DATA_DIR, "metadata.json")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

EXTRACT_METHOD = os.environ.get("CBIR_METHOD", "combined")
FIND_METRIC    = os.environ.get("CBIR_METRIC", "cosine")
LARAVEL_URL    = os.environ.get("LARAVEL_APP_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Lazy-loaded singletons
_extractor   = None
_finder      = None
_cbir_engine = None


def get_extractor_instance():
    global _extractor
    if _extractor is None:
        _extractor = get_extractor(EXTRACT_METHOD)
    return _extractor


def get_finder_instance():
    global _finder
    if _finder is None:
        _finder = get_finder(FIND_METRIC)
    return _finder


def get_engine() -> CBIREngine:
    """CBIREngine singleton — digunakan untuk /api/index/remove dan operasi lain."""
    global _cbir_engine
    if _cbir_engine is None:
        _cbir_engine = CBIREngine(database_path=DATA_DIR)
    return _cbir_engine


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def save_upload(file_or_bytes, filename: str) -> str:
    """Simpan file upload ke UPLOAD_FOLDER, return absolute path."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    safe_name = secure_filename(filename) or f"cbir-upload-{int(time.time())}.jpg"
    filepath  = os.path.join(UPLOAD_FOLDER, safe_name)
    if hasattr(file_or_bytes, "save"):
        file_or_bytes.save(filepath)
    else:
        with open(filepath, "wb") as f:
            f.write(file_or_bytes)
    return filepath


def compute_phash(image_path: str, hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) of an image, return hex string."""
    try:
        return str(imagehash.phash(Image.open(image_path), hash_size=hash_size))
    except Exception:
        return ""


def _reload_engine():
    """Reload CBIREngine setelah metadata.json diubah."""
    global _cbir_engine
    _cbir_engine = CBIREngine(database_path=DATA_DIR)


# ---------------------------------------------------------------------------
# Routes — Health & Status
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Wedding CBIR API",
        "version": "2.0.0",
        "status" : "running",
        "endpoints": {
            "GET  /health"                        : "Health check",
            "GET  /status"                        : "Database statistics",
            "GET  /api/index/stats"               : "Database statistics (alias)",
            "POST /api/search"                    : "Search similar images",
            "POST /api/index/add"                 : "Add image to index",
            "POST /api/index/remove"              : "Remove image from index",
            "POST /api/index/rebuild-from-dataset": "Rebuild index dari dataset.csv",
            "POST /api/index/clear"               : "Clear index",
            "POST /api/features/extract"          : "Extract features from image",
            "POST /api/sync"                      : "Trigger ai_sync.py (php artisan ai:sync)",
            "POST /api/arithmetic"                : "Aritmetika Citra: +, -, ×, ÷ antar gambar (AI fusion)",
            "GET  /api/arithmetic/ops"            : "Daftar operasi aritmetika yang tersedia",
        },
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status" : "healthy",
        "service": "Wedding CBIR API",
        "version": "2.0.0",
        "method" : EXTRACT_METHOD,
        "metric" : FIND_METRIC,
    })


@app.route("/status", methods=["GET"])
def status():
    try:
        db     = load_feature_database(FEATURE_DB)
        images = db.get("images", [])
        cats   = {}
        types  = {}
        for img in images:
            m = img.get("metadata", {})
            cats[m.get("category", "unknown")]  = cats.get(m.get("category", "unknown"), 0) + 1
            types[m.get("type", "unknown")]     = types.get(m.get("type", "unknown"), 0) + 1
        return jsonify({
            "status"        : "healthy",
            "total_products": len(images),
            "categories"    : cats,
            "types"         : types,
            "database_path" : FEATURE_DB,
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/index/stats", methods=["GET"])
def get_index_stats():
    return status()


# ---------------------------------------------------------------------------
# Routes — Search  (dipanggil CBIRService::searchByImage)
# ---------------------------------------------------------------------------

@app.route("/api/search", methods=["POST"])
def search_similar():
    """
    Cari gambar mirip.
    Dipanggil oleh Laravel CBIRService::searchByImage().

    Request : multipart/form-data  key='file'  (+ optional top_k)
              ATAU JSON {"image": "<base64>", "top_k": 10}
    Response: {"success": true, "results": [...], "query_time_seconds": 0.5}
    """
    t0 = time.perf_counter()

    try:
        # --- Terima gambar ---
        if "file" in request.files:
            f = request.files["file"]
            if not f or not f.filename:
                return jsonify({"error": "No file selected"}), 400
            if not allowed_file(f.filename):
                return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, bmp, webp"}), 400
            filepath = save_upload(f, f.filename)

        elif request.is_json and request.json and "image" in request.json:
            raw = request.json["image"]
            if "," in raw:
                raw = raw.split(",", 1)[1]
            img_bytes = base64.b64decode(raw)
            img       = Image.open(io.BytesIO(img_bytes))
            fname     = f"cbir-temp-{int(time.time())}.jpg"
            filepath  = save_upload(img_bytes, fname)
            img.save(filepath)

        else:
            return jsonify({"error": "No image provided"}), 400

        # --- top_k ---
        top_k_raw = request.form.get("top_k")
        if top_k_raw is None and request.is_json:
            top_k_raw = (request.json or {}).get("top_k")
        top_k = int(top_k_raw) if top_k_raw is not None else 10
        top_k = max(1, min(top_k, 50))

        # --- Ekstrak fitur & hitung skor ---
        import numpy as np
        extractor  = get_extractor_instance()
        finder     = get_finder_instance()
        query_feat = extractor.extract(filepath)

        db     = load_feature_database(FEATURE_DB)
        images = db.get("images", [])
        is_sim = finder.is_similarity()

        scores = []
        for entry in images:
            feat_dict = entry.get("features", {})
            feat_list = (
                feat_dict.get(EXTRACT_METHOD)
                or feat_dict.get("combined")
                or feat_dict.get("combined_features")
                or feat_dict.get("deep_features")
            )
            if feat_list is None:
                continue
            candidate = np.array(feat_list, dtype=np.float32)
            if candidate.shape != query_feat.shape:
                continue
            scores.append((finder.compute(query_feat, candidate), entry))

        scores.sort(key=lambda x: x[0], reverse=is_sim)

        # --- Re-ranking ensemble (post-processing) ---
        should_rerank = (request.form.get("rerank", "true").lower() == "true" or
                         (request.is_json and (request.json or {}).get("rerank", True)))
        if should_rerank and len(scores) > 1:
            try:
                reranked = rerank_by_ensemble(
                    query_emb=query_feat,
                    candidates=scores,
                    db_entries=images,
                    metric=FIND_METRIC,
                    top_k=top_k,
                )
                scores = [(entry.get("_rerank_score", s), entry)
                          for s, entry in scores[:len(reranked)]
                          if any(r.get("metadata", {}).get("owner_id") == entry.get("metadata", {}).get("owner_id")
                                 for r in reranked)]
                # Re-map from reranked list
                score_map = {}
                for entry in reranked:
                    oid = entry.get("metadata", {}).get("owner_id")
                    score_map[oid] = entry.get("_rerank_score", 0)
                scores = [(score_map.get(entry.get("metadata", {}).get("owner_id"), s), entry)
                          for s, entry in scores[:top_k]
                          if entry.get("metadata", {}).get("owner_id") in score_map]
            except Exception:
                pass

        # --- Format hasil (kompatibel dengan CBIRService.php) ---
        results = []
        for raw_score, entry in scores[:top_k]:
            meta      = entry.get("metadata", {})
            raw_score = float(raw_score)

            if is_sim:
                similarity = round(max(0.0, raw_score * 100.0), 2)
                score_01   = round(raw_score, 6)
            else:
                max_dist   = 25.0
                linear     = max(0.0, 100.0 - (raw_score / max_dist * 100.0))
                similarity = round((linear / 100.0) ** 2 * 100.0 if linear >= 15.0 else 0.0, 2)
                score_01   = round(similarity / 100.0, 6)

            item_type = meta.get("type", "product")
            if item_type not in ("product", "package"):
                item_type = "product"

            results.append({
                # Fields yang dibaca CBIRService.php
                "owner_id"      : meta.get("owner_id"),
                "type"          : item_type,
                "score"         : score_01,
                "similarity"    : similarity,
                "image_url"     : meta.get("image_url", ""),
                # Fields tambahan
                "id"            : entry.get("id"),
                "name"          : meta.get("name", ""),
                "category"      : meta.get("category", ""),
                "distance"      : round(raw_score, 4),
                "image_path"    : meta.get("image_path", ""),
                "price"         : meta.get("price", 0),
                "discount_price": meta.get("discount_price", 0),
            })

        elapsed = round(time.perf_counter() - t0, 3)

        return jsonify({
            "success"             : True,
            "results"             : results,
            "total_results"       : len(results),
            # Laravel CBIRService.php membaca "query_time_seconds"
            "query_time_seconds"  : elapsed,
            # Alias untuk kompatibilitas
            "query_time_s"        : elapsed,
            "method"              : EXTRACT_METHOD,
            "metric"              : FIND_METRIC,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — Index Management  (dipanggil CBIRService.php)
# ---------------------------------------------------------------------------

@app.route("/api/index/add", methods=["POST"])
def add_to_index():
    """
    Tambah satu gambar ke index.
    Dipanggil oleh Laravel CBIRService::indexMedia().

    Request JSON: {"image_path": "...", "metadata": {...}}
    """
    try:
        import numpy as np

        data       = request.json or {}
        image_path = data.get("image_path", "").strip()
        metadata   = data.get("metadata", {})

        if not image_path:
            return jsonify({"success": False, "message": "Missing image_path"}), 400
        
        image_path = resolve_portable_path(image_path)
        if not os.path.exists(image_path):
            return jsonify({"success": False, "message": f"File not found: {image_path}"}), 404

        # Validasi type
        if metadata.get("type") not in ("product", "package"):
            metadata["type"] = "product"

        if "image_path" in metadata:
            metadata["image_path"] = image_path

        extractor = get_extractor_instance()
        feat      = extractor.extract(image_path)
        phash_hex = compute_phash(image_path)

        db     = load_feature_database(FEATURE_DB)
        new_id = len(db["images"]) + 1

        # Simpan semua key fitur agar kompatibel dengan semua reader
        if hasattr(extractor, "_deep"):
            deep_feat    = extractor._deep.extract(image_path)
            color_feat   = extractor._color.extract(image_path)
            texture_feat = extractor._texture.extract(image_path)
        else:
            deep_feat    = feat if EXTRACT_METHOD in ("deep", "resnet50") else np.array([])
            color_feat   = feat if EXTRACT_METHOD in ("color", "color_histogram") else np.array([])
            texture_feat = feat if EXTRACT_METHOD == "lbp" else np.array([])

        entry = {
            "id"      : new_id,
            "path"    : image_path,
            "phash"   : phash_hex,
            "metadata": metadata,
            "features": {
                "combined"          : feat.tolist(),
                "combined_features" : feat.tolist(),
                "deep_features"     : deep_feat.tolist(),
                "color_histogram"   : color_feat.tolist(),
                "texture_features"  : texture_feat.tolist(),
                "deep"              : deep_feat.tolist(),
                "color"             : color_feat.tolist(),
                "method"            : EXTRACT_METHOD,
            },
        }
        db["images"].append(entry)
        save_feature_database(db, FEATURE_DB)
        _reload_engine()

        return jsonify({
            "success" : True,
            "message" : "Image added to index",
            "entry_id": new_id,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/index/remove", methods=["POST"])
def remove_from_index():
    """
    Hapus gambar dari index berdasarkan metadata_id atau owner_id+type.
    Dipanggil oleh Laravel CBIRService::removeFromIndex().

    Request JSON: {"metadata_id": 5}
                  ATAU {"owner_id": 5, "type": "product"}
    """
    try:
        data        = request.json or {}
        metadata_id = data.get("metadata_id")
        owner_id    = data.get("owner_id")
        item_type   = data.get("type", "product")

        db     = load_feature_database(FEATURE_DB)
        before = len(db["images"])

        if metadata_id is not None:
            # Hapus berdasarkan metadata.id (owner_id di metadata)
            db["images"] = [
                img for img in db["images"]
                if img.get("metadata", {}).get("owner_id") != int(metadata_id)
            ]
        elif owner_id is not None:
            # Hapus berdasarkan owner_id + type
            db["images"] = [
                img for img in db["images"]
                if not (
                    img.get("metadata", {}).get("owner_id") == int(owner_id)
                    and img.get("metadata", {}).get("type") == item_type
                )
            ]
        else:
            return jsonify({"success": False, "message": "Provide metadata_id or owner_id"}), 400

        # Re-number IDs
        for i, img in enumerate(db["images"], start=1):
            img["id"] = i

        removed = before - len(db["images"])
        save_feature_database(db, FEATURE_DB)
        _reload_engine()

        return jsonify({
            "success": True,
            "message": f"{removed} image(s) removed from index",
            "removed": removed,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/index/clear", methods=["POST"])
def clear_index():
    try:
        save_feature_database({"images": []}, FEATURE_DB)
        _reload_engine()
        return jsonify({"success": True, "message": "Index cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/index/rebuild-from-dataset", methods=["POST"])
def rebuild_from_dataset():
    """
    Rebuild index dari dataset.csv Laravel.
    Dipanggil oleh SyncCbirCsv.php dan SyncAICoreCommand.php.

    Request JSON (opsional): {"csv_path": "...", "app_url": "..."}
    """
    t0 = time.perf_counter()
    try:
        from src.features.build_features import build_features
        from ai_sync import sync_features_to_csv

        data     = request.json or {}
        csv_path = data.get("csv_path") or os.path.join(DATA_DIR, "dataset.csv")
        app_url  = data.get("app_url", LARAVEL_URL)

        if not os.path.exists(csv_path):
            return jsonify({
                "success": False,
                "message": f"Dataset CSV tidak ditemukan: {csv_path}. Jalankan php artisan cbir:sync.",
            }), 404

        # Build features → update metadata.json
        result = build_features(method="combined", csv_path=csv_path, app_url=app_url)

        # Sync fitur ke dataset.csv (angka muncul di CSV/Excel)
        sync_features_to_csv(csv_path=csv_path, metadata_path=FEATURE_DB)

        # Reload engine
        _reload_engine()

        combined = result.get("combined", {})
        cats     = {}
        types    = {}
        db       = load_feature_database(FEATURE_DB)
        for img in db.get("images", []):
            m = img.get("metadata", {})
            cats[m.get("category", "unknown")]  = cats.get(m.get("category", "unknown"), 0) + 1
            types[m.get("type", "unknown")]     = types.get(m.get("type", "unknown"), 0) + 1

        return jsonify({
            "success"        : True,
            "message"        : f"Rebuild selesai: {combined.get('total', 0)} gambar terindeks.",
            "total"          : combined.get("total", 0),
            "skipped"        : combined.get("skipped", 0),
            "errors"         : 0,
            "categories"     : cats,
            "types"          : types,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — AI Sync  (dipanggil php artisan ai:sync)
# ---------------------------------------------------------------------------

@app.route("/api/sync", methods=["POST"])
def trigger_sync():
    """
    Trigger ai_sync.py dari Laravel (php artisan ai:sync).
    Sama dengan rebuild-from-dataset tapi juga sync ke CSV.

    Request JSON (opsional): {"csv_path": "...", "app_url": "..."}
    """
    t0 = time.perf_counter()
    try:
        from ai_sync import main as ai_sync_main

        data     = request.json or {}
        csv_path = data.get("csv_path") or os.path.join(DATA_DIR, "dataset.csv")
        app_url  = data.get("app_url", LARAVEL_URL)

        exit_code = ai_sync_main(csv_path=csv_path, app_url=app_url)
        _reload_engine()

        if exit_code == 0:
            db    = load_feature_database(FEATURE_DB)
            total = len(db.get("images", []))
            return jsonify({
                "success"        : True,
                "message"        : f"AI Sync selesai: {total} gambar terindeks.",
                "total"          : total,
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
            })

        return jsonify({"success": False, "message": "AI Sync gagal. Cek log server."}), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — Feature Extraction (debugging)
# ---------------------------------------------------------------------------

@app.route("/api/features/extract", methods=["POST"])
def extract_features():
    """Extract fitur dari satu gambar (untuk debugging)."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        if not f or not f.filename or not allowed_file(f.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filepath  = save_upload(f, f.filename)
        extractor = get_extractor_instance()
        feat      = extractor.extract(filepath)

        return jsonify({
            "success"        : True,
            "method"         : EXTRACT_METHOD,
            "feature_dim"    : int(feat.shape[0]),
            "feature_preview": feat[:8].tolist(),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 16MB"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.route("/api/arithmetic", methods=["POST"])
def arithmetic_search_endpoint():
    """
    Aritmetika Citra: gabungkan 2+ gambar dengan operasi +, -, ×, ÷
    lalu cari hasilnya di database.

    Request:
      JSON: { "images": ["/path/gbr1.jpg", "/path/gbr2.jpg"], "operation": "...", ... }
      ATAU multipart: files image_1, image_2 + form fields operation, top_k, ...

    Response: { "success": true, "results": [...], "operation": {...}, ... }
    """
    t0 = time.perf_counter()
    try:
        images = []

        # --- Coba baca dari upload files ---
        f1 = request.files.get("image_1")
        f2 = request.files.get("image_2")
        if f1 and f1.filename:
            images.append(save_upload(f1, f1.filename))
        if f2 and f2.filename:
            images.append(save_upload(f2, f2.filename))

        # --- Kalau tidak ada file, baca dari JSON ---
        if not images:
            data = request.get_json(force=True) if request.is_json else {}
            raw = data.get("images", []) if isinstance(data, dict) else []
            if not isinstance(raw, list) or len(raw) < 1:
                return jsonify({"error": "Upload 2 file (image_1, image_2) atau kirim JSON 'images'"}), 400
            # Resolve portable path bila perlu
            for p in raw:
                resolved = resolve_portable_path(p) if not os.path.exists(p) else p
                images.append(resolved)

        if len(images) < 1:
            return jsonify({"error": "Minimal 1 gambar diperlukan"}), 400

        # --- Baca parameter ---
        json_data = request.get_json(silent=True) if request.is_json else {}
        if not isinstance(json_data, dict):
            json_data = {}
        operation = request.form.get("operation") or json_data.get("operation", "average")
        top_k = max(1, min(int(request.form.get("top_k") or json_data.get("top_k", 20)), 50))
        method = request.form.get("method") or json_data.get("method", EXTRACT_METHOD)
        metric = request.form.get("metric") or json_data.get("metric", FIND_METRIC)
        weights = json_data.get("weights")

        result = arithmetic_search(
            image_paths=images,
            operation=operation,
            method=method,
            metric=metric,
            top_k=top_k,
            db_path=FEATURE_DB,
            weights=weights,
        )
        result["server_time_seconds"] = round(time.perf_counter() - t0, 4)
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Arithmetic search failed: {str(e)}"}), 500


@app.route("/api/arithmetic/ops", methods=["GET"])
def arithmetic_ops():
    """Daftar operasi aritmetika yang tersedia."""
    return jsonify({
        "operations": {
            name: {"symbol": sym, "description": desc}
            for name, (_, sym, desc) in OPS.items()
        }
    })


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    host  = os.environ.get("FLASK_HOST", "0.0.0.0")
    port  = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print("=" * 55)
    print("  Wedding CBIR API Server v2.0")
    print("=" * 55)
    print(f"  Host      : {host}:{port}")
    print(f"  Method    : {EXTRACT_METHOD}")
    print(f"  Metric    : {FIND_METRIC}")
    print(f"  DB        : {FEATURE_DB}")
    print(f"  Laravel   : {LARAVEL_URL}")
    print(f"  Uploads   : {UPLOAD_FOLDER}")
    print("=" * 55)
    print("  Endpoints untuk Laravel:")
    print("    POST /api/search                     <- CBIRService::searchByImage()")
    print("    POST /api/index/add                  <- CBIRService::indexMedia()")
    print("    POST /api/index/remove               <- CBIRService::removeFromIndex()")
    print("    POST /api/index/rebuild-from-dataset <- SyncCbirCsv.php")
    print("    POST /api/sync                       <- php artisan ai:sync")
    print("    POST /api/arithmetic                 <- Aritmetika Citra (+, -, ×, ÷)")
    print("    GET  /api/arithmetic/ops             <- Daftar operasi aritmetika")
    print("    GET  /status                         <- CBIRController::getStats()")
    print("=" * 55)

    print("\nLoading extractor...")
    get_extractor_instance()
    print("Ready!\n")

    app.run(host=host, port=port, debug=debug, threaded=True)
