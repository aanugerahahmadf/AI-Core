import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR    = os.path.join(ROOT, "data")
DATASET_CSV = os.path.join(DATA_DIR, "dataset.csv")
METADATA    = os.path.join(DATA_DIR, "metadata.json")


class TestDataLoading(unittest.TestCase):
    def test_dataset_csv_exists(self):
        self.assertTrue(os.path.exists(DATASET_CSV), "dataset.csv tidak ditemukan. Jalankan: php artisan cbir:sync")

    def test_dataset_has_products_and_packages(self):
        from src.addons.data import load_dataset_csv
        rows = load_dataset_csv(DATASET_CSV)
        types = [r.get("Type","").lower() for r in rows]
        self.assertIn("product", types, "Tidak ada product di dataset.csv")
        self.assertIn("package", types, "Tidak ada package di dataset.csv")

    def test_dataset_no_item_type(self):
        from src.addons.data import load_dataset_csv
        rows = load_dataset_csv(DATASET_CSV)
        for row in rows:
            t = row.get("Type","").lower()
            self.assertNotEqual(t, "item", "Type 'item' tidak valid, harus 'product' atau 'package'")

    def test_metadata_json_exists(self):
        self.assertTrue(os.path.exists(METADATA), "metadata.json tidak ditemukan. Jalankan: python rebuild_index.py")

    def test_metadata_has_products_and_packages(self):
        from src.addons.data import load_feature_database
        db   = load_feature_database(METADATA)
        imgs = db.get("images", [])
        types = [img.get("metadata", {}).get("type", "") for img in imgs]
        self.assertIn("product", types, "Tidak ada product di metadata.json")
        self.assertIn("package", types, "Tidak ada package di metadata.json")

    def test_metadata_no_duplicates(self):
        from src.addons.data import load_feature_database
        db   = load_feature_database(METADATA)
        imgs = db.get("images", [])
        seen = set()
        for img in imgs:
            m   = img.get("metadata", {})
            key = str(m.get("type")) + "_" + str(m.get("owner_id"))
            self.assertNotIn(key, seen, "Duplikat ditemukan: " + key)
            seen.add(key)

    def test_metadata_no_empty_names(self):
        from src.addons.data import load_feature_database
        db   = load_feature_database(METADATA)
        imgs = db.get("images", [])
        for img in imgs:
            name = img.get("metadata", {}).get("name", "")
            self.assertTrue(len(name) > 0, "Name kosong untuk id=" + str(img.get("id")))

    def test_metadata_has_features(self):
        from src.addons.data import load_feature_database
        db   = load_feature_database(METADATA)
        imgs = db.get("images", [])
        for img in imgs:
            feat_dict = img.get("features", {})
            # Cek key "combined" (baru) atau "combined_features" (lama)
            feat = feat_dict.get("combined") or feat_dict.get("combined_features", [])
            self.assertEqual(len(feat), 2816, "combined/combined_features harus 2816-dim, id=" + str(img.get("id")))


class TestFinders(unittest.TestCase):
    def test_euclidean_finder(self):
        import numpy as np
        from src.addons.finder import get_finder
        finder = get_finder("euclidean")
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        dist = finder.compute(a, b)
        self.assertAlmostEqual(dist, 1.4142, places=3)

    def test_cosine_finder(self):
        import numpy as np
        from src.addons.finder import get_finder
        finder = get_finder("cosine")
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        sim = finder.compute(a, b)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_invalid_finder(self):
        from src.addons.finder import get_finder
        with self.assertRaises(ValueError):
            get_finder("invalid_metric")


class TestMetrics(unittest.TestCase):
    def test_map_perfect(self):
        from src.addons.metrics import mean_average_precision
        queries = [([1, 2, 3], {1, 2, 3})]
        self.assertAlmostEqual(mean_average_precision(queries), 1.0)

    def test_mrr(self):
        from src.addons.metrics import mean_reciprocal_rank
        queries = [([5, 1, 2], {1})]
        self.assertAlmostEqual(mean_reciprocal_rank(queries), 0.5)

    def test_first_rank_accuracy(self):
        from src.addons.metrics import first_rank_accuracy
        queries = [([1, 2], {1}), ([3, 1], {1})]
        self.assertAlmostEqual(first_rank_accuracy(queries), 0.5)


class TestCBIRSearch(unittest.TestCase):
    def test_metadata_loaded(self):
        from src.addons.data import load_feature_database
        db   = load_feature_database(METADATA)
        imgs = db.get("images", [])
        self.assertGreater(len(imgs), 0, "Index kosong. Jalankan: python -m src.features.build_features")

    def test_search_returns_results(self):
        """Cari gambar pertama di index sebagai query — harus dapat hasil."""
        from src.addons.data import load_feature_database
        from src.addons.extraction.extractor import get_extractor
        from src.addons.finder import get_finder
        import numpy as np

        db   = load_feature_database(METADATA)
        imgs = db.get("images", [])
        if not imgs:
            self.skipTest("Index kosong")

        # Ambil gambar pertama yang ada sebagai query
        query_path = None
        for img in imgs:
            p = img.get("path", "")
            if os.path.exists(p):
                query_path = p
                break

        if not query_path:
            self.skipTest("Tidak ada gambar yang bisa diakses")

        extractor = get_extractor("combined")
        finder    = get_finder("euclidean")
        query_vec = extractor.extract(query_path)

        scores = []
        for img in imgs:
            feat = img.get("features", {})
            vec  = feat.get("combined") or feat.get("combined_features")
            if vec:
                d = finder.compute(query_vec, np.array(vec, dtype="float32"))
                scores.append(d)

        self.assertGreater(len(scores), 0)
        # Gambar query sendiri harus jarak 0 (atau sangat kecil)
        self.assertAlmostEqual(min(scores), 0.0, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)