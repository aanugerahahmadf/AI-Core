# -*- coding: utf-8 -*-
"""
routes/context.py — Runtime bersama seluruh route AI Core.

Semua file route saling terhubung lewat modul ini:
  - path setup, .env, logging           (baris asli src/app.py)
  - konfigurasi: DATA_DIR/UPLOAD_FOLDER/FEATURE_DB, threshold CV, dst.
  - CBIREngine + lazy singletons (extractor, finder, engine)
  - helper CBIR (save_upload, phash, frame video) & CV/OCR (wajah, dokumen
    identitas KTP/SIM/NPWP/passport, bukti pembayaran)

Route memakai `from .context import *` (semua nama publik, persis
namespace monolit lama). Nama underscore diimpor eksplisit oleh route yang
memakainya sehingga ketidaksinkronan langsung terlihat sebagai ImportError.
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

# src/ is the package directory; project root (ai_core/) holds data/, logs/, .env
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)          # ai_core/src
BASE_DIR = os.path.dirname(ROOT_DIR)         # ai_core/ (project root)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Ensure logs directory exists
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging setup — write to ai_core/logs/ instead of project root
# ---------------------------------------------------------------------------

import logging
from logging import FileHandler

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        FileHandler(os.path.join(LOGS_DIR, "ai_service.out.log"), mode="a"),
        FileHandler(os.path.join(LOGS_DIR, "ai_service.err.log"), mode="a"),
    ],
)

# Suppress verbose werkzeug/development messages
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.WARNING)

# Silence the custom _log override if it was active
werkzeug.serving._log = original_log  # type: ignore

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from config import *  # noqa: F401,F403 - konfigurasi terpusat
from src.addons.data import load_feature_database, save_feature_database, resolve_portable_path
from src.addons.extraction.extractor import get_extractor
from src.addons.finder import get_finder
from src.addons.image_arithmetic import arithmetic_search, OPS
from src.addons.types import CATALOG, PACKAGE, normalize_type
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
        # Validasi type — hanya catalog atau package yang sah
        metadata["type"] = normalize_type(metadata.get("type"))
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
        logging.debug("[CV] mtcnn detect error: %s", e)
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
        logging.debug("[CV] OCR error: %s", e)
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

def _to_float(value):
    """Konversi expected_amount dari form/json (bisa '1500000.00')."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
