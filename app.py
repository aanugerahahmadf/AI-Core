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
  POST /api/face/verify               ← FaceService::verifyFace (verifikasi wajah vs KTP)
  POST /api/ktp/verify                ← FaceService::verifyKtp (validasi dokumen identitas
                                        KTP/SIM/NPWP/PASPORT + OCR nomor identitas)
  POST /api/proof/verify              ← FaceService::verifyProof (validasi bukti pembayaran)
"""

from __future__ import annotations

import base64
import io
import os
import re
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

    def search_similar_images(self, query_path):
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
        return results

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR      = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
FEATURE_DB    = os.path.join(DATA_DIR, "metadata.json")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "bmp", "webp"}
VIDEO_EXT     = {"mp4", "mov", "m4v", "webm", "3gp", "avi", "mkv"}
MAX_FILE_SIZE = 32 * 1024 * 1024  # 32 MB

EXTRACT_METHOD = os.environ.get("CBIR_METHOD", "combined")
FIND_METRIC    = os.environ.get("CBIR_METRIC", "cosine")
LARAVEL_URL    = os.environ.get("LARAVEL_APP_URL", "http://127.0.0.1:8000")

# Desimal yang disimpan untuk setiap komponen fitur vektor float.
FLOAT_PRECISION = 6

# --- Konfigurasi Computer Vision (verifikasi wajah & KTP) -------------------
FACE_VERIFY_THRESHOLD  = float(os.environ.get("FACE_VERIFY_THRESHOLD", "55"))
KTP_VERIFY_THRESHOLD   = float(os.environ.get("KTP_VERIFY_THRESHOLD", "50"))
FACE_CROP_SIZE         = int(os.environ.get("FACE_CROP_SIZE", "160"))
FACE_MIN_BLUR          = float(os.environ.get("FACE_MIN_BLUR", "25"))
FACE_MAX_FACES         = int(os.environ.get("FACE_MAX_FACES", "1"))
FACE_MIN_SIZE_RATIO    = float(os.environ.get("FACE_MIN_SIZE_RATIO", "0.05"))
FACE_MTCNN_PROB        = float(os.environ.get("FACE_MTCNN_PROB", "0.90"))

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


def is_video_file(filename: str) -> bool:
    """True bila ekstensi berkas adalah format video yang didukung."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in VIDEO_EXT


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


def extract_video_frame(video_path: str, max_frames: int = 120) -> str:
    """
    Ekstrak frame terbaik (paling tajam) dari video menjadi JPEG.

    Menggunakan cv2.VideoCapture (OpenCV + FFmpeg bawaan). Frame dipilih
    berdasarkan skor ketajaman (variance Laplacian); bila tidak ada frame
    yang terbaca, fallback ke frame pertama. Hasil disimpan di UPLOAD_FOLDER
    sebagai `video_frame_<timestamp>_<random>.jpg`.

    Raises ValueError bila video tidak bisa dibaca / tidak memiliki frame.
    """
    import cv2 as _cv

    cap = _cv.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError("Gagal membaca video")
        total = int(cap.get(_cv.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // max_frames) if total > max_frames else 1

        best_frame = None
        best_score = -1.0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                gray = _cv.cvtColor(frame, _cv.COLOR_BGR2GRAY)
                score = _cv.Laplacian(gray, _cv.CV_64F).var()
                if best_frame is None or score > best_score:
                    best_frame = frame
                    best_score = score
            idx += 1

        if best_frame is None:
            cap.set(_cv.CAP_PROP_POS_FRAMES, 0)
            ok, best_frame = cap.read()
        if best_frame is None:
            raise ValueError("Video tidak memiliki frame")

        out_path = os.path.join(
            UPLOAD_FOLDER,
            "video_frame_{}_{}.jpg".format(int(time.time()), os.urandom(3).hex()),
        )
        if not _cv.imwrite(out_path, best_frame):
            raise ValueError("Gagal menyimpan frame video")
        return out_path
    finally:
        cap.release()


def ensure_image_frame(filepath: str):
    """
    Normalisasi masukan menjadi gambar.

    Bila [filepath] adalah video, ekstrak frame terbaik dan kembalikan
    (frame_path, True). Bila sudah gambar, kembalikan (filepath, False).
    """
    if is_video_file(filepath):
        return extract_video_frame(filepath), True
    return filepath, False


def image_to_base64(path: str) -> str:
    """Encode berkas gambar menjadi base64 (tanpa prefix data URI)."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
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
            "POST /api/face/verify"               : "Verifikasi wajah (selfie vs KTP) — Computer Vision",
            "POST /api/ktp/verify"                : "Validasi dokumen KTP — Computer Vision",
            "POST /api/proof/verify"               : "Validasi bukti pembayaran — OCR (nominal)",
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
        "capabilities": {
            "cbir"           : True,
            "face_verify"    : True,
            "ktp_verify"     : True,
            "ocr"            : True,
            "proof_verify"   : True,
            "video_frames"   : True,
            "face_threshold" : FACE_VERIFY_THRESHOLD,
        },
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
        # --- top_k ---
        top_k_raw = request.form.get("top_k")
        if top_k_raw is None and request.is_json:
            top_k_raw = (request.json or {}).get("top_k")
        top_k = int(top_k_raw) if top_k_raw is not None else 10
        top_k = max(1, min(top_k, 50))

        # --- Terima gambar ---
        if "file" in request.files:
            f = request.files["file"]
            if not f or not f.filename:
                return jsonify({"error": "No file selected"}), 400
            if not allowed_file(f.filename) and not is_video_file(f.filename):
                return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, bmp, webp, mp4, mov, m4v, webm, 3gp"}), 400
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

        # --- Video: ekstrak frame terbaik agar bisa dianalisis CBIR ---
        filepath, is_frame = ensure_image_frame(filepath)

        # --- Ekstrak fitur & hitung skor ---
        import numpy as np
        extractor  = get_extractor_instance()
        finder     = get_finder_instance()
        query_feat = extractor.extract(filepath)
        # pHash query — dipakai untuk short-circuit "gambar sama persis" agar = 100%
        query_phash = compute_phash(filepath)

        db     = load_feature_database(FEATURE_DB)
        images = db.get("images", [])
        is_sim = finder.is_similarity()

        scores = []
        exact_hits = 0
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
            raw = finder.compute(query_feat, candidate)
            # Duplikat persis: pHash identik → skor maksimal (100%)
            stored_phash = entry.get("phash", "")
            if query_phash and stored_phash and query_phash == stored_phash:
                raw = 1.0 if is_sim else 0.0
                exact_hits += 1
            scores.append((raw, entry))

        if exact_hits:
            print(f"[CBIR] Exact duplicate(s) detected via pHash: {exact_hits}")

        scores.sort(key=lambda x: x[0], reverse=is_sim)

        # --- Dedup per (type, owner) + urutkan ulang (post-processing) ---
        # Skor asli (incl. short-circuit pHash = 100%) dipertahankan, hanya
        # memastikan satu hasil per item dan urutan menurun untuk similarity.
        should_rerank = (request.form.get("rerank", "true").lower() == "true" or
                         (request.is_json and (request.json or {}).get("rerank", True)))
        if should_rerank and len(scores) > 1:
            try:
                deduped: dict[str, tuple[float, dict]] = {}
                for s, entry in scores:
                    key = "{}_{}".format(
                        entry.get("metadata", {}).get("type", "product"),
                        entry.get("metadata", {}).get("owner_id"),
                    )
                    if key not in deduped:
                        deduped[key] = (s, entry)
                scores = list(deduped.values())
                scores.sort(key=lambda x: x[0], reverse=is_sim)
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
                "vendor"        : meta.get("vendor", ""),
                "description"   : meta.get("description", ""),
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
            "video_frame_extracted": is_frame,
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

        db = load_feature_database(FEATURE_DB)

        # Dedup per FILE, bukan per (type, owner_id).
        #
        # Sebuah produk/paket bisa punya banyak gambar (galeri). Kalau entri lain
        # milik (type, owner_id) yang sama ikut dihapus, setiap panggilan
        # `php artisan ai:sync` (yang looping SEMUA media per produk) akan
        # menyisakan hanya gambar terakhir per produk dan membuang sisanya —
        # indeks 400 entri bisa menyusut jadi 100 tanpa ada error sama sekali.
        #
        # Menampilkan "1 hasil per produk" memang tugas layer search
        # (`search_similar` dedup by type_owner_id, ambil skor terbaik), jadi di
        # sini cukup.replace entri ketika file yang sama di-index ulang.
        def _norm(p):
            return os.path.normcase(os.path.abspath(resolve_portable_path(p)))

        target_key = _norm(image_path)
        db["images"] = [img for img in db["images"] if _norm(img.get("path", "")) != target_key]

        existing_ids = [int(img.get("id") or 0) for img in db["images"]]
        new_id = (max(existing_ids) + 1) if existing_ids else 1

        # Simpan hanya key kanonik — alias fitur di-expand otomatis saat load,
        # sehingga tidak perlu diduplikasi di disk.
        if hasattr(extractor, "_deep"):
            deep_feat    = extractor._deep.extract(image_path)
            color_feat   = extractor._color.extract(image_path)
            texture_feat = extractor._texture.extract(image_path)
        else:
            deep_feat    = feat if EXTRACT_METHOD in ("deep", "resnet50") else np.array([])
            color_feat   = feat if EXTRACT_METHOD in ("color", "color_histogram") else np.array([])
            texture_feat = feat if EXTRACT_METHOD == "lbp" else np.array([])

        def _vec(arr):
            # WAJIB float64: float32 tidak bisa mewakili hasil round 6 desimal.
            return np.round(np.asarray(arr, dtype=np.float64), FLOAT_PRECISION).tolist()

        entry = {
            "id"      : new_id,
            "path"    : image_path,
            "phash"   : phash_hex,
            "metadata": metadata,
            "features": {
                "combined" : _vec(feat),
                "deep"     : _vec(deep_feat),
                "color"    : _vec(color_feat),
                "lbp"      : _vec(texture_feat),
                "method"   : EXTRACT_METHOD,
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
# Routes — Evaluation  (kualitatif: MAP, MRR, Precision@K, First Rank)
# ---------------------------------------------------------------------------

@app.route("/api/evaluate", methods=["GET"])
def evaluate():
    """
    Evaluasi kualitatif sistem CBIR.
    Menggunakan setiap gambar sebagai query dan membandingkan dengan ground truth
    (kategori yang sama = relevan).
    Response: { "success": true, "metrics": { "map": ..., "mrr": ..., ... } }
    """
    from src.evaluation.run_evaluation import run_and_save
    t0 = time.perf_counter()

    try:
        from src.addons.data import load_feature_database
        db     = load_feature_database(FEATURE_DB)
        images = db.get("images", [])

        if len(images) < 2:
            return jsonify({
                "success": False,
                "message": "Minimal 2 gambar diperlukan untuk evaluasi",
                "metrics": {},
            })

        # Evaluasi + persisten ke data/evaluation/
        payload   = run_and_save(db_path=FEATURE_DB, method=EXTRACT_METHOD, metric=FIND_METRIC)
        metrics   = payload["metrics"]
        n_queries = payload["n_queries"]
        elapsed   = round(time.perf_counter() - t0, 4)

        return jsonify({
            "success"                : True,
            "metrics"                : metrics,
            "precision_at_3"         : metrics.get("precision_at_3", 0.0),
            "n_images"               : len(images),
            "n_queries"              : n_queries,
            "evaluation_time_seconds": elapsed,
            "saved_paths"            : payload.get("saved_paths", []),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Computer Vision — Verifikasi Wajah & KTP
# (dipanggil FaceService.php di Laravel)
# ---------------------------------------------------------------------------

_face_cascade = None
_eye_cascade  = None
_face_encoder = None
_mtcnn        = None


def _get_face_cascade():
    """
    Haar cascade untuk deteksi wajah cepat.

    OpenCV 5.x menghapus API CascadeClassifier (dan modul cv2.objdetect), jadi
    yang dikembalikan None di sana — pemanggil harus memakai fallback MTCNN.
    """
    import cv2
    global _face_cascade
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    if _face_cascade is None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def _get_eye_cascade():
    import cv2
    global _eye_cascade
    if _eye_cascade is None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        _eye_cascade = cv2.CascadeClassifier(path)
    return _eye_cascade


def _get_face_encoder():
    """
    Deep embedding wajah — FaceNet InceptionResNetV1 (pretrained VGGFace2).
    Model khusus face recognition (seperti face_verify/3DiVi), menggantikan
    ResNet50-ImageNet lama yang bukan model wajah. Output embedding 512-d L2-normalized.
    Berat model diunduh otomatis pada pemakaian pertama.
    """
    global _face_encoder
    if _face_encoder is None:
        from facenet_pytorch import InceptionResnetV1
        _face_encoder = InceptionResnetV1(pretrained="vggface2", classify=False).eval()
    return _face_encoder


def _get_mtcnn():
    """
    Deteksi + aligment wajah via MTCNN (facenet_pytorch).
    Lebih akurat dari Haar cascade; menghasilkan crop 160x160 ternormalisasi [-1,1].
    """
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        _mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=False,
            select_largest=True,
            device="cpu",
        )
    return _mtcnn


def _detect_faces(image_path):
    """
    Deteksi wajah. Return (img_bgr, [ (x,y,w,h), ... ]).

    Memakai Haar cascade bila tersedia, jika tidak fallback ke MTCNN (OpenCV 5.x
    sudah tidak menyediakan CascadeClassifier).
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None, []

    cascade = _get_face_cascade()
    if cascade is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
        )
        return img, [tuple(int(v) for v in f) for f in faces]

    # Fallback MTCNN — mengembalikan (x, y, w, h) yang sama dengan Haar.
    try:
        _, boxes, _best, _tensor, _prob = _detect_faces_mtcnn(image_path)
    except Exception as e:
        print("[CV] Deteksi wajah gagal (Haar tidak tersedia & MTCNN error): " + str(e))
        return img, []
    return img, [tuple(int(v) for v in b) for b in boxes]


def _detect_faces_mtcnn(image_path):
    """
    Deteksi wajah via MTCNN (primary). Return:
    (img_bgr, all_boxes, best_box, aligned_tensor, best_prob)
      - all_boxes:    daftar semua (x, y, w, h) yang terdeteksi
      - best_box:     wajah terbesar/berconfidensi tertinggi atau None
      - aligned_tensor: tensor 160x160 [-1,1] crop wajah terbaik (untuk embedding)
    """
    import cv2
    import numpy as np
    img = cv2.imread(image_path)
    if img is None:
        return None, [], None, None, 0.0
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mtcnn = _get_mtcnn()
        boxes, probs = mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return img, [], None, None, 0.0
        boxes = [tuple(int(round(v)) for v in b) for b in boxes]
        best = int(np.argmax(probs))
        best_box = boxes[best]
        best_prob = float(probs[best])
        if best_prob < FACE_MTCNN_PROB:
            return img, [], None, None, best_prob
        aligned = mtcnn(rgb)
        if isinstance(aligned, (list, tuple)):
            aligned = aligned[0] if len(aligned) else None
        return img, boxes, best_box, aligned, best_prob
    except Exception as e:
        print("[CV] mtcnn detect error: " + str(e))
        return img, [], None, None, 0.0


def _eyes_open(img_bgr, box):
    """
    Anti-spoof: pastikan mata terbuka via Haar eye cascade di area wajah.
    Return (bool, int) — (mata terbuka, jumlah mata terdeteksi).
    """
    import cv2
    try:
        x, y, w, h = box
        cx0, cy0 = max(0, x + int(w * 0.15)), max(0, y + int(h * 0.25))
        cx1 = min(img_bgr.shape[1], x + int(w * 0.85))
        cy1 = min(img_bgr.shape[0], y + int(h * 0.55))
        region = img_bgr[cy0:cy1, cx0:cx1]
        if region.size == 0:
            return False, 0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        eyes = _get_eye_cascade().detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
        return len(eyes) > 0, len(eyes)
    except Exception:
        return True, 0


def _blur_score(gray):
    """Skor ketajaman gambar (varian Laplacian) — makin tinggi makin tajam."""
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _save_crop(image, face, path):
    """Potong & normalisasi wajah, simpan ke file, return path absolut."""
    import cv2
    x, y, w, h = face
    mx, my = int(w * 0.30), int(h * 0.30)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1 = min(image.shape[1], x + w + mx)
    y1 = min(image.shape[0], y + h + my)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
    if not cv2.imwrite(path, resized, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        return None
    return path


def _face_embedding(crop_path):
    """
    Embedding 512-d wajah (FaceNet) dari file crop.
    Crop diresize ke 160x160 & dinormalisasi ke [-1,1] sesuai standar FaceNet.
    """
    import numpy as np
    import torch
    try:
        img = Image.open(crop_path).convert("RGB").resize((160, 160))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            feat = _get_face_encoder()(tensor)
        return feat.squeeze().numpy().astype(np.float32)
    except Exception as e:
        print("[CV] face embedding error: " + str(e))
        return None


def _embedding_from_tensor(tensor):
    """Embedding 512-d dari tensor crop wajah 160x160 [-1,1] hasil MTCNN."""
    import numpy as np
    import torch
    try:
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            feat = _get_face_encoder()(tensor)
        return feat.squeeze().numpy().astype(np.float32)
    except Exception as e:
        print("[CV] tensor embedding error: " + str(e))
        return None


def _cosine_sim(a, b):
    import numpy as np
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _orb_match_ratio(path_a, path_b):
    """Rasio keypoint ORB yang cocok antar dua crop — skor 0..1."""
    import cv2
    try:
        orb = cv2.ORB_create(nfeatures=1000)
        img1 = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0.0
        _, des1 = orb.detectAndCompute(img1, None)
        _, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(des1, des2, k=2)
        good = 0
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good += 1
        return float(good / max(1, min(len(des1), len(des2))))
    except Exception:
        return 0.0


def _face_similarity(crop_a, crop_b, tensor_a=None, tensor_b=None):
    """
    Skor kemiripan dua wajah 0..100.
    Kombinasi embedding FaceNet (deep cosine) + keypoint ORB.
    Bila tensor aligned MTCNN tersedia, embedding dipakai dari tensor (lebih akurat).
    """
    emb_a = _embedding_from_tensor(tensor_a) if tensor_a is not None else _face_embedding(crop_a)
    emb_b = _embedding_from_tensor(tensor_b) if tensor_b is not None else _face_embedding(crop_b)
    deep  = _cosine_sim(emb_a, emb_b) if (emb_a is not None and emb_b is not None) else 0.0
    orb   = _orb_match_ratio(crop_a, crop_b)
    score = round(0.85 * deep * 100.0 + 0.15 * orb * 100.0, 2)
    return score, round(deep, 4), round(orb, 4)


# ---------------------------------------------------------------------------
# OCR (EasyOCR) — ekstraksi nomor identitas & nominal bukti pembayaran
# ---------------------------------------------------------------------------

KTP_NUMBER_PATTERN  = re.compile(r"(?<!\d)\d{16}(?!\d)")
PROOF_CURRENCY_RE   = re.compile(r"(?:Rp\.?\s*|IDR\.?\s*)?\d[\d.,]{2,}", re.IGNORECASE)
PROOF_OK_KEYWORDS   = ("TRANSFER", "BERHASIL", "SUKSES", "TERIMA", "DEBET", "INVOICE",
                       "PAYMENT", "BANK", "QRIS", "DIGITAL")

_ocr_reader = None


def _get_ocr_reader():
    """Lazy EasyOCR reader — model diunduh saat pemakaian pertama kali."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def _ocr_blocks(image_path):
    """OCR gambar. Return list (bbox, text, conf)."""
    try:
        return _get_ocr_reader().readtext(image_path, paragraph=False)
    except Exception as e:
        print("[CV] OCR error: " + str(e))
        return []


def _ocr_text(image_path):
    return [b[1] for b in _ocr_blocks(image_path)]


def _extract_ktp_number(texts):
    """Temukan Nomor KTP 16 digit dari hasil OCR."""
    joined = " ".join(texts)
    hit = KTP_NUMBER_PATTERN.search(joined)
    if hit:
        return hit.group(0)
    # Format terpotong dengan spasi antar grup (mis. "3171 0112 0303 0004")
    compact = re.sub(r"\s+", "", joined)
    hit = KTP_NUMBER_PATTERN.search(compact)
    if hit:
        # kembalikan versi asli 16 digit
        return hit.group(0)
    return None


# ---------------------------------------------------------------------------
# Deteksi tipe dokumen identitas (KTP / SIM / NPWP / PASSPORT)
# ---------------------------------------------------------------------------
# Cerminan scoreDocumentText() + regex dari mobile_app
# lib/core/utils/identity_document_utils/ agar skor AI Core dan mobile app
# memakai definisi dokumen yang sama.

IDENTITY_DOC_TYPES = ("ktp", "sim", "npwp", "passport")

# Nama resmi dokumen yang dipakai sebagai nilai `reason` pada /api/ktp/verify.
# `reason` HANYA boleh berisi salah satu dari empat nama ini (kontrak publik
# endpoint). Diagnosis kegagalan tidak lagi ditumpangkan ke `reason`; ia
# disimpan di `reason_code` (kode tunggal) dan `blocking_issue` (daftar lengkap).
IDENTITY_DOC_LABELS = {
    "ktp"     : "Kartu Tanda Penduduk",
    "sim"     : "Surat Izin Mengemudi",
    "npwp"    : "Nomor Pokok Wajib Pajak",
    "passport": "Passport",
}

# Pola nomor identitas per tipe dokumen.
# Lookaround PENTING: KTP 16 digit dan NPWP 15 digit berada di rentang yang
# berdekatan, jadi 15 digit TIDAK boleh diambil dari dalam string 16 digit.
_ID_NUMBER_DIGITS = {"ktp": 16, "npwp": 15, "sim": 12}

# Nomor paspor alfanumerik (mis. "C1234567"); sisanya dibaca dari MRZ.
_ID_PASSPORT_PATTERN = re.compile(r"\b[A-Z]{1,2}[0-9]{6,8}\b")

# Baris MRZ (Machine Readable Zone) paspor: P<INDSUMANDRA<<JANE<ANNE<
_MRZ_LINE_RE   = re.compile(r"P<[A-Z]{3}[A-Z<]{10,}")
_MRZ_NUMBER_RE = re.compile(r"[A-Z0-9]{6,9}")

# Kata kunci OCR per tipe dokumen, dipaired dengan bobotnya.
_ID_DOC_KEYWORDS = {
    "ktp": (
        ("NIK", 3), ("NAMA", 1), ("ALAMAT", 1), ("PENDIDIKAN", 1),
        ("KEWARGANEGARAAN", 1), ("PROVINSI", 1), ("BERLAKU HINGGA", 1),
        ("GOLONGAN DARAH", 1), ("TEMPAT/TGL LAHIR", 1), ("AGAMA", 1),
        ("STATUS PERKAWINAN", 1), ("PEKERJAAN", 1), ("KECAMATAN", 1),
        ("KELURAHAN", 1),
    ),
    "sim": (
        ("SURAT IZIN MENGEMUDI", 3), ("GOLONGAN", 1), ("NO. SIM", 1),
        ("BERLAKU", 1),
    ),
    "npwp": (
        ("NPWP", 3), ("WAJIB PAJAK", 1), ("KARTU PAJAK", 1),
    ),
    "passport": (
        ("PASSPORT", 2), ("REPUBLIK INDONESIA", 1),
        ("REPUBLIC OF INDONESIA", 1),
    ),
}

# Bonus skor bila pola nomor identitas tipe tersebut cocok.
_ID_NUMBER_BONUS = {"ktp": 3, "npwp": 3, "sim": 1, "passport": 0}


def _id_doc_max_score(doc_type: str) -> int:
    """Skor maksimum yang mungkin dicapai untuk satu tipe dokumen."""
    return sum(w for _, w in _ID_DOC_KEYWORDS[doc_type]) + _ID_NUMBER_BONUS[doc_type]


def _find_digit_run(upper: str, count: int):
    """
    Cari tepat `count` digit berurutan pada `upper`.

    Digit boleh dipisah satu spasi/titik/strip di antaranya, karena NPWP
    Indonesia dicetak bergaris "09.254.294.3-407.000" dan SIM "1234 5678 9012".

    Lookaround di kiri/kanan memastikan jumlah digitnya persis — inilah yang
    mencegah 15 digit diambil dari dalam NIK 16 digit.
    """
    body = r"(?:\d[\s.\-]?)"
    pattern = re.compile(
        r"(?<![\d.])(" + body + "{" + str(count) + r"," + str(count * 2) + r"})(?![\d.])"
    )
    for match in pattern.finditer(upper):
        digits = re.sub(r"\D", "", match.group(1))
        if len(digits) == count:
            return digits
    return None


def _extract_identity_number(doc_type: str, upper: str, compact: str):
    """
    Ambil nomor identitas sesuai tipe dokumen dari teks OCR.

    Args:
        doc_type: 'ktp' | 'sim' | 'npwp' | 'passport'
        upper   : teks OCR yang sudah di-uppercase.
        compact : teks OCR tanpa spasi (cadangan).
    """
    digit_count = _ID_NUMBER_DIGITS.get(doc_type)
    if digit_count is not None:
        for haystack in (upper, compact):
            found = _find_digit_run(haystack, digit_count)
            if found:
                return found
        return None

    # Paspor: pola alfanumerik, lalu fallback ke baris MRZ.
    for haystack in (upper, compact):
        hit = _ID_PASSPORT_PATTERN.search(haystack)
        if hit:
            return hit.group(0)

    for mrz in _MRZ_LINE_RE.findall(upper):
        hit = _MRZ_NUMBER_RE.search(mrz)
        if hit:
            return hit.group(0)

    return None


def _score_identity_document(doc_type: str, upper: str, compact: str) -> int:
    """Skor kecocokan teks OCR terhadap satu tipe dokumen."""
    score = 0
    for keyword, weight in _ID_DOC_KEYWORDS[doc_type]:
        if keyword in upper:
            score += weight

    if doc_type == "passport":
        if _MRZ_LINE_RE.search(upper):
            score += 3
    elif _extract_identity_number(doc_type, upper, compact):
        score += _ID_NUMBER_BONUS[doc_type]

    return score


def _aspect_score(aspect, doc_type: str = "ktp"):
    """Skor kesesuaian rasio aspek dokumen identitas — 0..100."""
    if doc_type == "passport":
        # Halaman data paspor lebih lebar dari kartu ID-1.
        if 1.15 <= aspect <= 1.55:
            return 100.0
        if 1.05 <= aspect <= 1.75:
            return 70.0
        return 30.0

    # KTP / SIM / NPWP semuanya format kartu ID-1 (CR80: 85.60 x 53.98 mm).
    if 1.50 <= aspect <= 1.72:
        return 100.0
    if 1.40 <= aspect <= 1.85:
        return 70.0
    return 30.0


def _detect_identity_document(texts, aspect, hint: str | None = None):
    """
    Deteksi tipe dokumen identitas dari hasil OCR.

    Args:
        texts: List string hasil OCR.
        aspect: Rasio aspek gambar (w/h).
        hint  : Tipe dokumen yang diharapkan client (opsional). Bila diisi,
                tipe itu dikembalikan apa adanya — skor semua tipe tetap
                dikembalikan supaya ketidakcocokan tetap terlihat.

    Returns:
        (doc_type|None, scores, numbers)
        - doc_type: 'ktp' | 'sim' | 'npwp' | 'passport', atau None bila tidak ada yang cocok.
        - scores  : dict tipe -> skor keywords + nomor.
        - numbers : dict tipe -> nomor identitas terbaca (atau None).
    """
    upper   = " ".join(texts).upper()
    compact = re.sub(r"\s+", "", upper)

    scores  = {t: _score_identity_document(t, upper, compact) for t in IDENTITY_DOC_TYPES}
    numbers = {t: _extract_identity_number(t, upper, compact) for t in IDENTITY_DOC_TYPES}

    # Hint client: dump ke tipe yang diminta, skoring tetap dihitung untuk semua.
    if hint in IDENTITY_DOC_TYPES:
        return hint, scores, numbers

    # Auto-detect: skor ternormalisasi, dengan bonus bila rasio aspek cocok.
    best, best_norm = None, 0.0
    for doc_type, score in scores.items():
        maximum = _id_doc_max_score(doc_type)
        if maximum <= 0 or score <= 0:
            continue
        normalized = score / maximum
        if _aspect_score(aspect, doc_type) >= 100.0:
            normalized *= 1.15
        if normalized > best_norm:
            best, best_norm = doc_type, normalized

    return best, scores, numbers


def _best_identity_doc_type(scores):
    """
    Tipe dokumen dengan skor keyword tertinggi, atau None bila semua nol.

    Dipakai sebagai fallback terakhir penentuan `reason` di /api/ktp/verify saat
    OCR tidak menghasilkan kecocokan yang cukup untuk dianggap dokumen yang
    dikenal, tetapi `reason` tetap harus berisi salah satu dari empat nama
    dokumen.
    """
    best, best_score = None, 0
    for doc_type in IDENTITY_DOC_TYPES:
        score = scores.get(doc_type, 0)
        if score > best_score:
            best, best_score = doc_type, score
    return best


def _normalize_number(raw):
    """Konversi string angka Indonesia (Rp 1.500.000 / 1500000 / 1,500,000) ke angka."""
    s = re.sub(r"[^\d.,]", "", raw)
    if not s:
        return None
    comma = s.count(",")
    dot = s.count(".")
    if comma and dot:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "").replace(".", "")
    elif comma:
        if comma >= 2 or len(s.rsplit(",", 1)[1]) == 3:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif dot:
        if dot >= 2 or len(s.rsplit(".", 1)[1]) == 3:
            s = s.replace(".", "")
        else:
            s = s.replace(".", ".")
    try:
        return float(s)
    except ValueError:
        try:
            return float("".join(ch for ch in s if ch.isdigit()))
        except ValueError:
            return None


def _extract_amounts(texts):
    """Ekstrak nominal (>= 100) dari hasil OCR bukti pembayaran."""
    amounts = []
    for text in texts:
        for m in PROOF_CURRENCY_RE.finditer(text):
            val = _normalize_number(m.group(0))
            if val is not None and val >= 100.0:
                amounts.append(int(round(val)))
    return sorted(set(amounts), reverse=True)


@app.route("/api/face/verify", methods=["POST"])
def face_verify():
    """
    Verifikasi identitas via wajah.
    Dipanggil oleh Laravel FaceService::verifyFace().

    Request (multipart): selfie=<file> [+ reference=<file/KTP>]
    ATAU JSON: {"selfie_path": "...", "reference_path": "..."}

    Response: {
      "success": true,
      "verified": bool,
      "face_detected_selfie": bool,
      "faces_selfie": int,
      "face_detected_reference": bool|null,
      "similarity": float|null,
      "deep_similarity": float|null,
      "orb_ratio": float|null,
      "threshold": float,
      "reason": "MATCH|NO_MATCH|NO_FACE_IN_SELFIE|NO_FACE_IN_REFERENCE|NEEDS_REFERENCE|BLURRY|MULTIPLE_FACES|FACE_TOO_SMALL|EYES_CLOSED",
      "blur_score_selfie": float,
      "eyes_open_selfie": bool|null,
      "liveness_checks": object|null
    }
    """
    t0 = time.perf_counter()
    try:
        def _resolve(key):
            if key in request.files and request.files[key] and request.files[key].filename:
                f = request.files[key]
                if not allowed_file(f.filename) and not is_video_file(f.filename):
                    return None, "Invalid file type"
                return save_upload(f, f.filename), None
            if request.is_json:
                p = ((request.json or {}).get(key + "_path") or "").strip()
                if p:
                    resolved = resolve_portable_path(p)
                    if os.path.exists(resolved):
                        return resolved, None
            return None, None

        selfie_path, err = _resolve("selfie")
        if err:
            return jsonify({"success": False, "error": err}), 400
        if not selfie_path:
            return jsonify({"success": False, "error": "Field 'selfie' diperlukan"}), 400

        reference_path, rerr = _resolve("reference")
        if rerr:
            return jsonify({"success": False, "error": rerr}), 400

        # Video: ekstrak frame terbaik dari selfie & referensi
        selfie_path, selfie_was_video = ensure_image_frame(selfie_path)
        reference_was_video = False
        if reference_path:
            reference_path, reference_was_video = ensure_image_frame(reference_path)

        img_selfie, faces_s_all, best_s, tensor_s, prob_s = _detect_faces_mtcnn(selfie_path)
        if img_selfie is None:
            return jsonify({"success": False, "error": "Gambar selfie tidak valid"}), 400

        # Fallback: bila MTCNN tidak menemukan wajah, coba Haar cascade
        if not best_s:
            _, haar_faces = _detect_faces(selfie_path)
            if haar_faces:
                faces_s_all = haar_faces
                best_s = max(haar_faces, key=lambda f: f[2] * f[3])
                tensor_s = None
                prob_s = 0.0

        import cv2
        gray_selfie = cv2.cvtColor(img_selfie, cv2.COLOR_BGR2GRAY)
        blur_selfie = round(_blur_score(gray_selfie), 2)

        n_faces_selfie = len(faces_s_all)

        payload = {
            "success"                : True,
            "verified"               : False,
            "face_detected_selfie"   : best_s is not None,
            "faces_selfie"           : n_faces_selfie,
            "face_detected_reference": None,
            "faces_reference"        : 0,
            "similarity"             : None,
            "deep_similarity"        : None,
            "orb_ratio"              : None,
            "threshold"              : FACE_VERIFY_THRESHOLD,
            "reason"                 : "",
            "blur_score_selfie"      : blur_selfie,
            "eyes_open_selfie"       : None,
            "liveness_checks"        : None,
            "query_time_seconds"     : round(time.perf_counter() - t0, 3),
        }

        if selfie_was_video:
            payload["frame_base64_selfie"] = image_to_base64(selfie_path)
        if reference_was_video:
            payload["frame_base64_reference"] = image_to_base64(reference_path)

        if not best_s:
            payload["reason"] = "NO_FACE_IN_SELFIE"
            return jsonify(payload)

        # --- Anti-spoof server-side ---
        if blur_selfie < FACE_MIN_BLUR:
            payload["reason"] = "BLURRY"
            return jsonify(payload)

        if n_faces_selfie > FACE_MAX_FACES:
            payload["reason"] = "MULTIPLE_FACES"
            return jsonify(payload)

        h_s, w_s = img_selfie.shape[:2]
        face_ratio = max(best_s[2] / float(w_s), best_s[3] / float(h_s))
        if face_ratio < FACE_MIN_SIZE_RATIO:
            payload["reason"] = "FACE_TOO_SMALL"
            return jsonify(payload)

        eyes_open, n_eyes = _eyes_open(img_selfie, best_s)
        payload["eyes_open_selfie"] = eyes_open
        if not eyes_open:
            payload["reason"] = "EYES_CLOSED"
            return jsonify(payload)

        payload["liveness_checks"] = {
            "blur_pass"        : blur_selfie >= FACE_MIN_BLUR,
            "single_face"      : n_faces_selfie == 1,
            "face_size_ok"     : face_ratio >= FACE_MIN_SIZE_RATIO,
            "eyes_open"        : True,
            "eyes_detected"    : n_eyes,
            "mtcnn_confidence" : round(prob_s, 4),
        }

        if not reference_path:
            payload["reason"] = "NEEDS_REFERENCE"
            return jsonify(payload)

        img_ref, faces_ref, best_r, tensor_r, _ = _detect_faces_mtcnn(reference_path)
        if img_ref is None:
            return jsonify({"success": False, "error": "Gambar referensi tidak valid"}), 400

        if not best_r:
            _, haar_ref = _detect_faces(reference_path)
            if haar_ref:
                faces_ref = haar_ref
                best_r = max(haar_ref, key=lambda f: f[2] * f[3])
                tensor_r = None

        payload["face_detected_reference"] = best_r is not None
        payload["faces_reference"] = len(faces_ref)

        if not best_r:
            payload["reason"] = "NO_FACE_IN_REFERENCE"
            return jsonify(payload)

        crop_s = _save_crop(img_selfie, best_s, os.path.join(UPLOAD_FOLDER, "face_selfie.png"))
        crop_r = _save_crop(img_ref, best_r, os.path.join(UPLOAD_FOLDER, "face_ref.png"))
        if not crop_s or not crop_r:
            payload["reason"] = "CROP_ERROR"
            return jsonify(payload)

        similarity, deep_sim, orb_ratio = _face_similarity(crop_s, crop_r, tensor_s, tensor_r)
        payload["similarity"]      = similarity
        payload["deep_similarity"] = deep_sim
        payload["orb_ratio"]       = orb_ratio
        payload["verified"]        = similarity >= FACE_VERIFY_THRESHOLD
        payload["reason"]          = "MATCH" if payload["verified"] else "NO_MATCH"
        payload["query_time_seconds"] = round(time.perf_counter() - t0, 3)

        return jsonify(payload)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/ktp/verify", methods=["POST"])
def ktp_verify():
    """
    Validasi dokumen identitas (KTP / SIM / NPWP / PASSPORT) via Computer Vision.
    Dipanggil oleh Laravel FaceService::verifyKtp().

    Request (multipart): image=<file>   ATAU JSON {"image_path": "..."}
        Field opsional `doc_type` / JSON `doc_type`:
            'ktp' | 'sim' | 'npwp' | 'passport'
            Bila diisi, tipe itu yang dilaporkan (client sudah tahu dokumen apa
            yang sedang dipindai). Bila kosong, tipe dideteksi otomatis dari
            teks OCR + rasio aspek.

    Response: {
      "success": true,
      "verified": bool,
      "score": float,
      "reason": "Kartu Tanda Penduduk|Surat Izin Mengemudi|Passport|Nomor Pokok Wajib Pajak",
      "reason_code": "NO_FACE|NO_TEXT|WRONG_ASPECT|BLURRY|UNKNOWN_DOCUMENT"|null,
      "document_type": "ktp|sim|npwp|passport"|null,
      "document_number": string|null,
      "document_scores": {"ktp": int, "sim": int, "npwp": int, "passport": int},
      "is_ktp_like": bool,
      "face_detected": bool,
      "aspect_ratio": float,
      "blur_score": float,
      ...
    }

    CATATAN: `reason` SELALU berisi nama resmi dokumen yang dianalisis --
    hanya ada empat nilai yang mungkin ("Kartu Tanda Penduduk",
    "Surat Izin Mengemudi", "Passport", "Nomor Pokok Wajib Pajak"), apa pun
    hasil validasinya. Penyebab kegagalan tidak hilang: ada di `reason_code`
    (kode tunggal berurutan prioritas) dan `blocking_issue` (daftar lengkap).
    """
    t0 = time.perf_counter()
    try:
        # -- doc_type hint dari client (opsional) --
        raw_hint = request.form.get("doc_type")
        if raw_hint is None and request.is_json:
            raw_hint = (request.json or {}).get("doc_type")
        doc_hint = str(raw_hint or "").strip().lower() or None
        if doc_hint is not None and doc_hint not in IDENTITY_DOC_TYPES:
            return jsonify({
                "success": False,
                "error": "doc_type harus salah satu dari: " + ", ".join(IDENTITY_DOC_TYPES),
            }), 400

        if "image" in request.files and request.files["image"] and request.files["image"].filename:
            f = request.files["image"]
            if not allowed_file(f.filename) and not is_video_file(f.filename):
                return jsonify({"success": False, "error": "Invalid file type"}), 400
            image_path = save_upload(f, f.filename)
        elif request.is_json and (request.json or {}).get("image_path"):
            image_path = resolve_portable_path(request.json["image_path"])
            if not os.path.exists(image_path):
                return jsonify({"success": False, "error": "File tidak ditemukan"}), 404
        else:
            return jsonify({"success": False, "error": "Field 'image' diperlukan"}), 400

        # Video: ekstrak frame terbaik sebelum analisis dokumen
        image_path, was_video = ensure_image_frame(image_path)

        img, faces = _detect_faces(image_path)
        if img is None:
            return jsonify({"success": False, "error": "Gambar tidak valid"}), 400

        import cv2
        h, w = img.shape[:2]
        aspect = round(w / float(h), 4)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = round(_blur_score(gray), 2)

        largest = max(faces, key=lambda f: f[2] * f[3]) if faces else None
        face_w_ratio = (largest[2] / float(w)) if largest else 0.0
        has_face = largest is not None and 0.05 <= face_w_ratio <= 0.60

        # OCR sekali, dipakai untuk deteksi tipe + nomor + skor keywords
        texts          = []
        doc_type       = None
        doc_scores     = {t: 0 for t in IDENTITY_DOC_TYPES}
        doc_numbers    = {t: None for t in IDENTITY_DOC_TYPES}
        doc_number     = None
        try:
            texts = _ocr_text(image_path)
            doc_type, doc_scores, doc_numbers = _detect_identity_document(texts, aspect, doc_hint)
            if doc_type:
                doc_number = doc_numbers.get(doc_type)
        except Exception as e:
            print("[CV] OCR warning: " + str(e))

        has_text      = any(len(t.strip()) >= 3 for t in texts)
        number_found = doc_number is not None

        # Rasio aspect skor dihitung sesuai tipe dokumen (paspor beda dari kartu ID-1)
        aspect_s = _aspect_score(aspect, doc_type or doc_hint or "ktp")
        face_s   = 100.0 if has_face else 0.0
        blur_s   = min(100.0, (blur / 80.0) * 100.0)
        text_s   = min(100.0, (doc_scores.get(doc_type, 0) / _id_doc_max_score(doc_type)) * 100.0) if doc_type else 0.0

        # --- Masalah yang memblokir validasi (syarat mutlak, bukan cuma skor) ---
        blocking = []
        if not has_face:
            blocking.append("NO_FACE")
        if aspect_s < 70.0:
            blocking.append("WRONG_ASPECT")
        if blur_s < 35.0:
            blocking.append("BLURRY")
        if doc_type is None:
            blocking.append("UNKNOWN_DOCUMENT")

        score = round(
            0.35 * aspect_s + 0.25 * face_s + 0.15 * blur_s + 0.25 * text_s
            + (10.0 if number_found else 0.0),
            2,
        )
        # Tanpa syarat mutlak ini, foto tanpa wajah / aspect salah tetap lolos
        # hanya karena skor aspect+blur+nomor yang tinggi.
        verified = bool(not blocking and score >= KTP_VERIFY_THRESHOLD)

        # --- reason = nama resmi dokumen, SELALU salah satu dari 4 tipe ---
        # Rantai fallback: hasil deteksi OCR -> hint client -> tipe dengan skor
        # keyword tertinggi -> default "ktp". Tidak ada nilai diagnostik di sini
        # karena `reason` adalah kontrak nama dokumen, bukan kode error.
        reason_doc = doc_type or doc_hint or _best_identity_doc_type(doc_scores) or "ktp"
        reason     = IDENTITY_DOC_LABELS[reason_doc]

        # Kode diagnostik (tetap satu nilai, prioritas berurutan) untuk consumer
        # yang butuh alasan kegagalan tanpa membaca `blocking_issue`.
        if not has_face:
            reason_code = "NO_FACE"
        elif not has_text:
            reason_code = "NO_TEXT"
        elif "WRONG_ASPECT" in blocking:
            reason_code = "WRONG_ASPECT"
        elif "BLURRY" in blocking:
            reason_code = "BLURRY"
        elif "UNKNOWN_DOCUMENT" in blocking:
            reason_code = "UNKNOWN_DOCUMENT"
        else:
            reason_code = None

        is_card_id1 = (doc_type or doc_hint) != "passport"

        return jsonify({
            "success"           : True,
            "verified"          : bool(verified),
            "score"             : score,
            "reason"            : reason,
            "reason_code"       : reason_code,
            "blocking_issue"    : blocking or None,
            # --- tipe dokumen & nomor identitas ---
            "document_type"     : doc_type,
            "document_type_hint": doc_hint,
            "document_number"   : doc_number,
            "document_numbers"  : {t: v for t, v in doc_numbers.items() if v},
            "document_scores"   : doc_scores,
            "document_score_max": _id_doc_max_score(doc_type) if doc_type else 0,
            # --- metrik CV (backward compatible) ---
            "is_ktp_like"       : bool(has_face and aspect_s >= 70.0 and is_card_id1),
            "is_identity_doc"   : bool(has_face and aspect_s >= 70.0),
            "face_detected"     : bool(has_face),
            "faces_detected"    : len(faces),
            "aspect_ratio"      : aspect,
            "aspect_score"      : aspect_s,
            "blur_score"        : blur,
            "text_detected"     : bool(has_text),
            "number_detected"   : bool(number_found),
            "ktp_number"        : doc_number if doc_type == "ktp" else None,
            "ktp_number_detected": bool(doc_type == "ktp" and number_found),
            "threshold"         : KTP_VERIFY_THRESHOLD,
            "query_time_seconds": round(time.perf_counter() - t0, 3),
            "frame_base64"      : image_to_base64(image_path) if was_video else "",
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Computer Vision — Verifikasi Bukti Pembayaran (OCR nominal)
# ---------------------------------------------------------------------------

@app.route("/api/proof/verify", methods=["POST"])
def proof_verify():
    """
    Validasi bukti pembayaran via OCR + Computer Vision.
    Dipanggil oleh Laravel FaceService::verifyProof().

    Request (multipart): image=<file> [+ expected_amount=<angka>]
    ATAU JSON: {"image_path": "...", "expected_amount": 1500000}

    Response: {
      "success": true,
      "verified": bool,
      "amounts": [int, ...],
      "matched_amount": int|null,
      "expected_amount": float|null,
      "has_text": bool,
      "blur_score": float,
      "reason": "OK|NO_AMOUNT|NO_MATCH_TOTAL|BLURRY|NO_TEXT"
    }
    """
    t0 = time.perf_counter()
    try:
        expected = None
        if request.is_json and (request.json or {}).get("image_path"):
            image_path = resolve_portable_path(request.json["image_path"])
            if not os.path.exists(image_path):
                return jsonify({"success": False, "error": "File tidak ditemukan"}), 404
            expected = _to_float((request.json or {}).get("expected_amount"))
        elif "image" in request.files and request.files["image"] and request.files["image"].filename:
            f = request.files["image"]
            if not allowed_file(f.filename) and not is_video_file(f.filename):
                return jsonify({"success": False, "error": "Invalid file type"}), 400
            image_path = save_upload(f, f.filename)
            expected = _to_float(request.form.get("expected_amount"))
        else:
            return jsonify({"success": False, "error": "Field 'image' diperlukan"}), 400

        # Video: ekstrak frame terbaik sebelum OCR bukti pembayaran
        image_path, was_video = ensure_image_frame(image_path)

        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"success": False, "error": "Gambar tidak valid"}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = round(_blur_score(gray), 2)
        blur_s = min(100.0, (blur / 60.0) * 100.0)

        texts = _ocr_text(image_path)
        has_text = any(len(t.strip()) >= 3 for t in texts)
        amounts = _extract_amounts(texts)
        blob = " ".join(x.upper() for x in texts)
        text_signal = any(k in blob for k in PROOF_OK_KEYWORDS)

        matched = None
        if expected is not None:
            for cand in amounts:
                if abs(cand - expected) <= 500:
                    matched = cand
                    break

        if expected is not None:
            verified = matched is not None
            if not amounts:
                reason = "NO_AMOUNT"
            elif not verified:
                reason = "NO_MATCH_TOTAL"
            elif blur_s < 30.0:
                reason = "BLURRY"
            else:
                reason = "OK"
        else:
            verified = bool(amounts) and blur_s >= 30.0
            if not amounts:
                reason = "NO_AMOUNT"
            elif blur_s < 30.0:
                reason = "BLURRY"
            elif not has_text:
                reason = "NO_TEXT"
            else:
                reason = "OK"

        return jsonify({
            "success"           : True,
            "verified"          : bool(verified),
            "amounts"           : amounts[:10],
            "matched_amount"    : matched,
            "expected_amount"   : expected,
            "has_text"          : bool(has_text),
            "has_payment_signal": bool(text_signal),
            "blur_score"        : blur,
            "reason"            : reason,
            "query_time_seconds": round(time.perf_counter() - t0, 3),
            "frame_base64"      : image_to_base64(image_path) if was_video else "",
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _to_float(value):
    """Konversi expected_amount dari form/json (bisa '1500000.00')."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
      ATAU multipart: files image_1, image_2 + form fields operation, ...

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
        method = request.form.get("method") or json_data.get("method", EXTRACT_METHOD)
        metric = request.form.get("metric") or json_data.get("metric", FIND_METRIC)
        weights = json_data.get("weights")

        result = arithmetic_search(
            image_paths=images,
            operation=operation,
            method=method,
            metric=metric,
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
    print("    POST /api/face/verify                <- FaceService::verifyFace()")
    print("    POST /api/ktp/verify                 <- FaceService::verifyKtp()")
    print("    POST /api/proof/verify               <- FaceService::verifyProof()")
    print("    GET  /api/arithmetic/ops             <- Daftar operasi aritmetika")
    print("    GET  /api/evaluate                   <- Evaluasi kualitatif (MAP, MRR, dll)")
    print("    GET  /status                         <- CBIRController::getStats()")
    print("=" * 55)

    print("\nLoading extractor...")
    get_extractor_instance()
    print("Ready!\n")

    app.run(host=host, port=port, debug=debug, threaded=True)
