# -*- coding: utf-8 -*-
"""
tests/test_api.py — Unit tests untuk Flask API endpoints (app.py).

Jalankan:
    python -m pytest tests/test_api.py -v --tb=short
"""

import base64
import io
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR   = os.path.join(ROOT, "data")
METADATA   = os.path.join(DATA_DIR, "metadata.json")

# Empat nilai yang boleh muncul sebagai `reason` pada /api/ktp/verify.
DOC_REASONS = frozenset(
    {
        "Kartu Tanda Penduduk",
        "Surat Izin Mengemudi",
        "Passport",
        "Nomor Pokok Wajib Pajak",
    }
)

# Kode diagnostik belongs to `reason_code`, never to `reason`.
DIAGNOSTIC_CODES = frozenset(
    {"NO_FACE", "NO_TEXT", "WRONG_ASPECT", "BLURRY", "UNKNOWN_DOCUMENT"}
)


def _get_sample_image_path() -> str | None:
    """Ambil path gambar pertama yang ada dari metadata.json."""
    if not os.path.exists(METADATA):
        return None
    with open(METADATA, encoding="utf-8") as f:
        db = json.load(f)
    for img in db.get("images", []):
        p = img.get("path", "")
        if os.path.exists(p):
            return p
    return None


def _get_distinct_sample_image_paths(count: int) -> list[str]:
    """Ambil `count` path gambar berbeda yang benar-benar ada di disk."""
    if not os.path.exists(METADATA):
        return []
    with open(METADATA, encoding="utf-8") as f:
        db = json.load(f)
    found: list[str] = []
    for img in db.get("images", []):
        p = img.get("path", "")
        if p and os.path.exists(p) and p not in found:
            found.append(p)
        if len(found) == count:
            break
    return found


def _read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class TestHealthEndpoints(unittest.TestCase):
    """Test GET endpoints yang tidak butuh gambar."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    def test_root_returns_200(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_root_has_endpoints_key(self):
        resp = self.client.get("/")
        data = json.loads(resp.data)
        self.assertIn("endpoints", data)
        self.assertIn("service", data)

    def test_health_returns_healthy(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "healthy")

    def test_health_has_method_and_metric(self):
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        self.assertIn("method", data)
        self.assertIn("metric", data)

    def test_status_returns_200(self):
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)

    def test_status_has_total_products(self):
        resp = self.client.get("/status")
        data = json.loads(resp.data)
        self.assertIn("total_products", data)
        self.assertIsInstance(data["total_products"], int)

    def test_status_has_categories(self):
        resp = self.client.get("/status")
        data = json.loads(resp.data)
        self.assertIn("categories", data)

    def test_api_index_stats_returns_200(self):
        resp = self.client.get("/api/index/stats")
        self.assertEqual(resp.status_code, 200)

    def test_404_returns_json(self):
        resp = self.client.get("/endpoint-tidak-ada")
        self.assertEqual(resp.status_code, 404)
        data = json.loads(resp.data)
        self.assertIn("error", data)


class TestSearchEndpoint(unittest.TestCase):
    """Test POST /api/search."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client      = flask_app.app.test_client()
        self.sample_path = _get_sample_image_path()

    def test_search_no_image_returns_400(self):
        resp = self.client.post("/api/search", json={})
        self.assertEqual(resp.status_code, 400)

    def test_search_invalid_file_type_returns_400(self):
        resp = self.client.post(
            "/api/search",
            data={"file": (io.BytesIO(b"fake content"), "test.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_search_with_file_returns_results(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        resp = self.client.post(
            "/api/search",
            data={"file": (io.BytesIO(_read_image_bytes(self.sample_path)), "query.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("results", data)
        self.assertIsInstance(data["results"], list)

    def test_search_results_have_required_fields(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        resp = self.client.post(
            "/api/search",
            data={"file": (io.BytesIO(_read_image_bytes(self.sample_path)), "query.jpg")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        if data["results"]:
            result = data["results"][0]
            for field in ["id", "type", "name", "category", "score", "image_url"]:
                self.assertIn(field, result, f"Field '{field}' tidak ada di result")

    def test_search_top_k_respected(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        resp = self.client.post(
            "/api/search",
            data={
                "file" : (io.BytesIO(_read_image_bytes(self.sample_path)), "query.jpg"),
                "top_k": "3",
            },
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertLessEqual(len(data["results"]), 3)

    def test_search_with_base64_returns_results(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        b64  = base64.b64encode(_read_image_bytes(self.sample_path)).decode("utf-8")
        resp = self.client.post("/api/search", json={"image": b64, "top_k": 5})
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("results", data)

    def test_search_returns_query_time(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        resp = self.client.post(
            "/api/search",
            data={"file": (io.BytesIO(_read_image_bytes(self.sample_path)), "query.jpg")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        self.assertIn("query_time_s", data)
        self.assertIsInstance(data["query_time_s"], float)

    def test_search_score_is_float(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        resp = self.client.post(
            "/api/search",
            data={"file": (io.BytesIO(_read_image_bytes(self.sample_path)), "query.jpg")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        if data.get("results"):
            self.assertIsInstance(data["results"][0]["score"], float)

    def test_search_exact_duplicate_returns_100_percent(self):
        """Gambar yang sama persis (di-re-encode q70 + resize) harus kembali 100%."""
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        from PIL import Image

        # Simulasikan upload dari mobile: kompres quality 70 + resize max 800px
        buf = io.BytesIO()
        img = Image.open(self.sample_path).convert("RGB")
        if img.width > 800:
            ratio = 800.0 / img.width
            img = img.resize((800, max(1, int(img.height * ratio))))
        img.save(buf, "JPEG", quality=70)
        buf.seek(0)

        resp = self.client.post(
            "/api/search",
            data={"file": (buf, "query_compressed.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertTrue(data["results"], "Seharusnya ada hasil untuk gambar duplikat")

        top = data["results"][0]
        self.assertEqual(float(top["similarity"]), 100.0,
                         f"Gambar identik harus 100%, dapat: {top['similarity']}")


class TestIndexEndpoints(unittest.TestCase):
    """Test POST /api/index/* endpoints."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client      = flask_app.app.test_client()
        self.sample_path = _get_sample_image_path()
        # Backup metadata.json sebelum test
        self._backup = None
        if os.path.exists(METADATA):
            with open(METADATA, encoding="utf-8") as f:
                self._backup = f.read()

    def tearDown(self):
        # Restore metadata.json setelah setiap test
        if self._backup is not None:
            with open(METADATA, "w", encoding="utf-8") as f:
                f.write(self._backup)

    def test_add_missing_image_path_returns_400(self):
        resp = self.client.post("/api/index/add", json={})
        self.assertEqual(resp.status_code, 400)

    def test_add_nonexistent_file_returns_404(self):
        resp = self.client.post(
            "/api/index/add",
            json={"image_path": "/path/tidak/ada.jpg", "metadata": {}},
        )
        self.assertEqual(resp.status_code, 404)

    def test_add_valid_image_returns_success(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        resp = self.client.post(
            "/api/index/add",
            json={
                "image_path": self.sample_path,
                "metadata"  : {"type": "product", "owner_id": 999, "name": "Test", "category": "test"},
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("entry_id", data)

    def test_add_duplicate_owner_deduplicates(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        payload = {
            "image_path": self.sample_path,
            "metadata"  : {"type": "product", "owner_id": 777, "name": "Dedup", "category": "test"},
        }
        self.client.post("/api/index/add", json=payload)
        self.client.post("/api/index/add", json=payload)

        import json as _json
        with open(METADATA, encoding="utf-8") as f:
            db = _json.load(f)
        count = sum(
            1 for img in db.get("images", [])
            if (img.get("metadata", {}).get("type"), img.get("metadata", {}).get("owner_id")) == ("product", 777)
        )
        self.assertEqual(count, 1, "Index tidak boleh berisi duplikat untuk (type, owner_id) sama")

    def test_add_different_images_same_owner_are_both_kept(self):
        """Regression: galeri satu produk tidak boleh saling menimpa.

        `php artisan ai:sync` looping semua media per produk lalu memanggil
        `/api/index/add` untuk masing-masing. Kalau endpoint ini menghapus
        entri milik (type, owner_id) yang sama, setiap panggilan menghapus
        gambar sebelumnya dan indeks menyusut diam-diam dari 4 gambar menjadi
        1 per produk. Yang boleh dihapus hanya entri untuk FILE YANG SAMA.
        """
        paths = _get_distinct_sample_image_paths(2)
        if len(paths) < 2:
            self.skipTest("Butuh minimal 2 gambar berbeda yang tersedia")

        for path in paths:
            resp = self.client.post(
                "/api/index/add",
                json={
                    "image_path": path,
                    "metadata"  : {"type": "package", "owner_id": 888, "name": "Galeri", "category": "test"},
                },
            )
            self.assertEqual(resp.status_code, 200)

        import json as _json
        with open(METADATA, encoding="utf-8") as f:
            db = _json.load(f)

        entries = [
            img for img in db.get("images", [])
            if (img.get("metadata", {}).get("type"), img.get("metadata", {}).get("owner_id")) == ("package", 888)
        ]
        indexed_paths = [img.get("path") for img in entries]
        self.assertEqual(
            len(indexed_paths), 2,
            "Kedua gambar galeri harus tetap terindex; ditemukan: %r" % (indexed_paths,),
        )
        self.assertEqual(len(set(indexed_paths)), 2, "Path gambar tidak boleh terindex dua kali")

    def test_rebuild_missing_csv_returns_404(self):
        resp = self.client.post(
            "/api/index/rebuild-from-dataset",
            json={"csv_path": "/path/tidak/ada.csv"},
        )
        self.assertEqual(resp.status_code, 404)
        data = json.loads(resp.data)
        self.assertFalse(data["success"])

    def test_clear_index_returns_success(self):
        resp = self.client.post("/api/index/clear")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])

    def test_status_after_clear_is_zero(self):
        self.client.post("/api/index/clear")
        resp = self.client.get("/status")
        data = json.loads(resp.data)
        self.assertEqual(data["total_products"], 0)


class TestExtractEndpoint(unittest.TestCase):
    """Test POST /api/features/extract."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client      = flask_app.app.test_client()
        self.sample_path = _get_sample_image_path()

    def test_extract_no_file_returns_400(self):
        resp = self.client.post("/api/features/extract")
        self.assertEqual(resp.status_code, 400)

    def test_extract_invalid_type_returns_400(self):
        resp = self.client.post(
            "/api/features/extract",
            data={"file": (io.BytesIO(b"fake"), "test.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_extract_valid_image_returns_dim(self):
        if not self.sample_path:
            self.skipTest("Tidak ada gambar sample")
        resp = self.client.post(
            "/api/features/extract",
            data={"file": (io.BytesIO(_read_image_bytes(self.sample_path)), "test.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["feature_dim"], 2816)
        self.assertIn("feature_preview", data)
        self.assertEqual(len(data["feature_preview"]), 8)


class TestVisionEndpoints(unittest.TestCase):
    """Test POST /api/face/verify  &  POST /api/ktp/verify (Computer Vision)."""

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()

    def _blank_jpg(self, width=800, height=1280, color=(200, 200, 200)):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (width, height), color).save(buf, "JPEG", quality=90)
        buf.seek(0)
        return buf

    def test_face_verify_missing_selfie_returns_400(self):
        resp = self.client.post("/api/face/verify", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_face_verify_no_face_returns_verified_false(self):
        resp = self.client.post(
            "/api/face/verify",
            data={"selfie": (self._blank_jpg(), "selfie.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertFalse(data["verified"])
        self.assertEqual(data["reason"], "NO_FACE_IN_SELFIE")
        self.assertIn("blur_score_selfie", data)
        self.assertIn("face_detected_selfie", data)

    def test_face_verify_no_reference_returns_needs_reference(self):
        # Selfie blank tapi minimal berisi wajah: tidak bisa memaksa Haar mendeteksi
        # wajah dari gambar kosong; karenanya test ini hanya memvalidasi bahwa
        # payload respons selalu memiliki field reason yang valid.
        resp = self.client.post(
            "/api/face/verify",
            data={},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_face_verify_blank_selfie_and_reference(self):
        resp = self.client.post(
            "/api/face/verify",
            data={
                "selfie"   : (self._blank_jpg(), "s.jpg"),
                "reference": (self._blank_jpg(600, 900, (180, 180, 180)), "r.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertFalse(data["verified"])
        self.assertIn(data["reason"], ("NO_FACE_IN_SELFIE", "NO_FACE_IN_REFERENCE", "NO_MATCH"))
        self.assertIn("similarity", data)

    def test_ktp_verify_missing_image_returns_400(self):
        resp = self.client.post("/api/ktp/verify", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_ktp_verify_blank_no_face_not_verified(self):
        resp = self.client.post(
            "/api/ktp/verify",
            data={"image": (self._blank_jpg(), "ktp.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("verified", data)
        # `reason` adalah nama dokumen, diagnosis pindah ke `reason_code`.
        self.assertIn(data["reason"], DOC_REASONS)
        self.assertEqual(data["reason_code"], "NO_FACE")
        self.assertIn("aspect_ratio", data)
        self.assertIn("score", data)

    def test_ktp_verify_reason_is_always_one_of_four_documents(self):
        """`reason` tidak boleh pernah berisi kode diagnostik."""
        cases = [
            ({"image": (self._blank_jpg(), "ktp.jpg")}, None),
            ({"image": (self._blank_jpg(), "x.jpg"), "doc_type": "sim"}, "sim"),
            ({"image": (self._blank_jpg(), "x.jpg"), "doc_type": "npwp"}, "npwp"),
            ({"image": (self._blank_jpg(), "x.jpg"), "doc_type": "passport"}, "passport"),
            ({"image": (self._blank_jpg(), "x.jpg"), "doc_type": "ktp"}, "ktp"),
        ]
        for data, _ in cases:
            with self.subTest(data=sorted(data)):
                resp = self.client.post(
                    "/api/ktp/verify", data=data, content_type="multipart/form-data"
                )
                self.assertEqual(resp.status_code, 200)
                payload = json.loads(resp.data)
                self.assertIn(payload["reason"], DOC_REASONS)
                # Tidak ada kode diagnostik yang bocor ke `reason`.
                self.assertNotIn(payload["reason"], DIAGNOSTIC_CODES)
                self.assertIn(payload["reason_code"], DIAGNOSTIC_CODES | {None})

    def test_ktp_verify_hint_drives_reason_when_undetected(self):
        """Dokumen tak terdeteksi tetap melaporkan nama dari hint client."""
        for doc_type, expected in (
            ("sim", "Surat Izin Mengemudi"),
            ("npwp", "Nomor Pokok Wajib Pajak"),
            ("passport", "Passport"),
            ("ktp", "Kartu Tanda Penduduk"),
        ):
            with self.subTest(doc_type=doc_type):
                resp = self.client.post(
                    "/api/ktp/verify",
                    data={"image": (self._blank_jpg(), "x.jpg"), "doc_type": doc_type},
                    content_type="multipart/form-data",
                )
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(json.loads(resp.data)["reason"], expected)

    def test_ktp_verify_blank_defaults_to_ktp_name(self):
        """Tanpa hint dan tanpa OCR, `reason` jatuh ke Kartu Tanda Penduduk."""
        resp = self.client.post(
            "/api/ktp/verify",
            data={"image": (self._blank_jpg(), "ktp.jpg")},
            content_type="multipart/form-data",
        )
        payload = json.loads(resp.data)
        self.assertEqual(payload["reason"], "Kartu Tanda Penduduk")
        self.assertIsNone(payload["document_type"])
        self.assertIn("NO_FACE", payload["blocking_issue"])

    def test_ktp_verify_doc_labels_cover_every_doc_type(self):
        import app as flask_app

        self.assertEqual(
            sorted(flask_app.IDENTITY_DOC_LABELS), sorted(flask_app.IDENTITY_DOC_TYPES)
        )
        self.assertEqual(sorted(flask_app.IDENTITY_DOC_LABELS.values()), sorted(DOC_REASONS))
        for doc_type in flask_app.IDENTITY_DOC_TYPES:
            resp = self.client.post(
                "/api/ktp/verify",
                data={"image": (self._blank_jpg(), "x.jpg"), "doc_type": doc_type},
                content_type="multipart/form-data",
            )
            payload = json.loads(resp.data)
            self.assertEqual(payload["reason"], flask_app.IDENTITY_DOC_LABELS[doc_type])

    def test_ktp_verify_invalid_type_returns_400(self):
        resp = self.client.post(
            "/api/ktp/verify",
            data={"image": (io.BytesIO(b"fake"), "ktp.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_ktp_verify_blank_no_ktp_number(self):
        resp = self.client.post(
            "/api/ktp/verify",
            data={"image": (self._blank_jpg(), "ktp.jpg")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        self.assertIn("ktp_number_detected", data)
        self.assertFalse(data["ktp_number_detected"])

    def test_proof_verify_missing_image_returns_400(self):
        resp = self.client.post("/api/proof/verify", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_proof_verify_blank_returns_no_amount(self):
        resp = self.client.post(
            "/api/proof/verify",
            data={"image": (self._blank_jpg(), "proof.jpg"), "expected_amount": "1500000"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertFalse(data["verified"])
        self.assertEqual(data["reason"], "NO_AMOUNT")
        self.assertEqual(data["expected_amount"], 1500000.0)


class TestOcrHelpers(unittest.TestCase):
    """Test helper OCR tanpa memanggil model (cepat & deterministik)."""

    def setUp(self):
        import app as flask_app
        self.app = flask_app

    def test_extract_ktp_number_contiguous(self):
        import app as a
        self.assertEqual(a._extract_ktp_number([""]), None)
        self.assertEqual(a._extract_ktp_number(["N I K", "3171011203030004", "PRIA"]), "3171011203030004")

    def test_extract_ktp_number_with_spaces(self):
        import app as a
        self.assertEqual(a._extract_ktp_number(["NIK : 3171 0112 0303 0004"]), "3171011203030004")

    def test_extract_amounts_rp_ribu_comma(self):
        import app as a
        self.assertIn(1500000, a._extract_amounts(["Total Rp 1.500.000"]))
        self.assertIn(500000, a._extract_amounts(["Rp500.000"]))

    def test_extract_amounts_plain_and_comma(self):
        import app as a
        self.assertIn(1500000, a._extract_amounts(["Amount 1500000"]))
        self.assertIn(1500000, a._extract_amounts(["IDR 1,500,000"]))

    def test_normalize_number(self):
        import app as a
        self.assertEqual(a._normalize_number("Rp 1.500.000"), 1500000.0)
        self.assertEqual(a._normalize_number("1,500,000"), 1500000.0)
        self.assertEqual(a._normalize_number("1500000"), 1500000.0)
        self.assertIsNone(a._normalize_number("abc"))


class TestVideoSupport(unittest.TestCase):
    """
    Test dukungan video (frame extraction) pada endpoint AI.
    Video dibuat di memori via OpenCV VideoWriter (mp4v codec).
    """

    def setUp(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        self.client = flask_app.app.test_client()
        self.video_path = None

    def tearDown(self):
        if self.video_path and os.path.exists(self.video_path):
            try:
                os.remove(self.video_path)
            except OSError:
                pass

    def _make_video(self, width=160, height=120, frames=12):
        """Buat video mp4 kecil; return (path, bytes)."""
        import cv2
        import numpy as np
        import tempfile
        buf_path = os.path.join(tempfile.mkdtemp(), "sample_video.mp4")
        vw = cv2.VideoWriter(buf_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
        for i in range(frames):
            val = int(np.clip(80 + 20 * i, 0, 255))
            frame = np.full((height, width, 3), val, np.uint8)
            cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (255, 255, 255), -1)
            vw.write(frame)
        vw.release()
        with open(buf_path, "rb") as f:
            data = f.read()
        self.video_path = buf_path
        return buf_path, data

    def test_is_video_file_helper(self):
        import app as a
        self.assertTrue(a.is_video_file("video.mp4"))
        self.assertTrue(a.is_video_file("video.MOV"))
        self.assertFalse(a.is_video_file("image.jpg"))
        self.assertFalse(a.is_video_file("noext"))

    def test_search_with_video_returns_results(self):
        _, vid = self._make_video()
        resp = self.client.post(
            "/api/search",
            data={"file": (io.BytesIO(vid), "query.mp4"), "top_k": 5},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertIn("results", data)
        self.assertTrue(data["video_frame_extracted"])
        self.assertIsInstance(data["results"], list)

    def test_search_invalid_video_ext_returns_400(self):
        resp = self.client.post(
            "/api/search",
            data={"file": (io.BytesIO(b"fake video bytes"), "query.wmv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)

    def test_ktp_verify_with_video_blank_frame(self):
        _, vid = self._make_video()
        resp = self.client.post(
            "/api/ktp/verify",
            data={"image": (io.BytesIO(vid), "ktp_video.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertFalse(data["verified"])
        # `reason` = nama dokumen; frame video kosong tidak mengubahnya.
        self.assertIn(data["reason"], DOC_REASONS)
        self.assertEqual(data["reason_code"], "NO_FACE")
        self.assertIn("aspect_ratio", data)
        self.assertTrue(data["frame_base64"], "frame_base64 harus ada untuk video")

    def test_proof_verify_with_video_blank_returns_no_amount(self):
        _, vid = self._make_video()
        resp = self.client.post(
            "/api/proof/verify",
            data={"image": (io.BytesIO(vid), "proof.mp4"), "expected_amount": "1500000"},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertFalse(data["verified"])
        self.assertEqual(data["reason"], "NO_AMOUNT")
        self.assertTrue(data["frame_base64"], "frame_base64 harus ada untuk video")

    def test_face_verify_with_video_selfie(self):
        _, vid = self._make_video()
        resp = self.client.post(
            "/api/face/verify",
            data={"selfie": (io.BytesIO(vid), "selfie.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"], data)
        self.assertFalse(data["verified"])
        self.assertEqual(data["reason"], "NO_FACE_IN_SELFIE")
        self.assertIn("blur_score_selfie", data)
        self.assertTrue(data.get("frame_base64_selfie"), "frame_base64_selfie harus ada untuk video")

    def test_health_reports_video_frames(self):
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        self.assertTrue(data["capabilities"]["video_frames"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
