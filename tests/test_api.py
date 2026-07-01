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


if __name__ == "__main__":
    unittest.main(verbosity=2)
