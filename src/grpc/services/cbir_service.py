"""
AI Core gRPC Service Implementation
Wedding Flower Decoration App - CBIR gRPC Server
"""

import os
import sys
import time
import tempfile
from concurrent import futures
import grpc

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT PROTOBUF — JANGAN memakai stub kosong.
#
# Versi sebelumnya menulis `from generated import cbir_pb2` setelah menambah
# `src/grpc/generated` ke sys.path. Import itu SELALU gagal (tidak ada package
# bernama `generated` di dalam folder generated), lalu except-nya menelan
# kegagalan itu dan mendefinisikan `class cbir_pb2: pass`. Akibatnya:
#
#   - server tetap start dan port 50052 tetap LISTENING,
#   - tapi setiap method yang butuh membuat respons melempar
#     "type object 'cbir_pb2' has no attribute 'CbirSearchResponse'",
#   - yang di sisi klien terlihat sebagai gRPC code 2 (UNKNOWN), bukan
#     "server mati" — persis gejala yang sulit dilacak.
#
# Sekarang: utamakan import relatif dari package, fallback ke path absolut
# `src/grpc` (dipakai kalau modul diimpor sebagai `services.cbir_service`),
# dan kalau keduanya gagal, GAGALKAN import dengan pesan jelas — lebih baik
# server tidak pernah start daripada start dalam keadaan tidak bisa menjawab.
# ─────────────────────────────────────────────────────────────────────────────
try:  # diimpor sebagai src.grpc.services.cbir_service
    from ..generated import cbir_pb2
    from ..generated import cbir_pb2_grpc
except ImportError:  # diimpor sebagai services.cbir_service
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from generated import cbir_pb2  # type: ignore[no-redef]
    from generated import cbir_pb2_grpc  # type: ignore[no-redef]

from src.addons.data import load_feature_database, save_feature_database, resolve_portable_path
from src.addons.extraction.extractor import get_extractor
from src.addons.finder import get_finder
from src.addons.types import normalize_type
from src.config.settings import DATA_DIR, METADATA_PATH
import numpy as np


class CbirServiceServicer:
    """gRPC Service Implementation for CBIR operations"""

    def __init__(self, data_dir=None, feature_db_path=None):
        # Path default HARUS menunjuk ke `ai_core/data`, bukan `ai_core/src/data`.
        # Versi sebelumnya memakai `dirname(__file__)/../../data`, dan dari
        # `src/grpc/services/` itu berarti `src/data` -- folder yang hanya
        # berisi `uploads/` dan `__init__.py`. Akibatnya server gRPC selalu
        # melaporkan `total_images: 0` dan pencarian selalu kosong, walaupun
        # `data/metadata.json` (20 MB, berisi seluruh fitur) ada di root
        # proyek. Sumber kebenaran path-nya ada di src/config/settings.py.
        default_data_dir = str(DATA_DIR)
        default_metadata = str(METADATA_PATH)

        self.data_dir = data_dir or default_data_dir
        self.feature_db_path = feature_db_path or default_metadata
        self.upload_folder = os.path.join(self.data_dir, 'uploads')
        os.makedirs(self.upload_folder, exist_ok=True)
        
        self._extractor = None
        self._finder = None

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = get_extractor('combined')
        return self._extractor

    @property
    def finder(self):
        if self._finder is None:
            self._finder = get_finder('cosine')
        return self._finder

    def Search(self, request, context):
        """Search for similar images using CBIR"""
        start_time = time.time()
        
        try:
            if not request.image_data:
                return self._create_search_response(
                    results=[], search_time=0, total_results=0, error="No image data provided"
                )

            # Save uploaded image temporarily
            temp_file = tempfile.NamedTemporaryFile(
                dir=self.upload_folder, suffix='.jpg', delete=False
            )
            temp_file.write(request.image_data)
            temp_file.close()

            try:
                # Extract features from query image
                query_features = self.extractor.extract(temp_file.name)
                
                # Load database
                database = load_feature_database(self.feature_db_path)
                images = database.get('images', [])
                
                # Compute similarities
                limit = request.limit if request.limit > 0 else 10
                min_similarity = request.min_similarity
                
                results = []
                for entry in images:
                    features = entry.get('features', {})
                    feature_list = (
                        features.get('combined') or
                        features.get('combined_features') or
                        features.get('deep_features')
                    )
                    
                    if feature_list is None:
                        continue
                    
                    candidate = np.array(feature_list, dtype=np.float32)
                    if candidate.shape != query_features.shape:
                        continue
                    
                    similarity = self.finder.compute(query_features, candidate)
                    
                    # Convert to percentage
                    if self.finder.is_similarity():
                        similarity_pct = max(0.0, similarity * 100.0)
                    else:
                        max_dist = 25.0
                        linear = max(0.0, 100.0 - (similarity / max_dist * 100.0))
                        similarity_pct = (linear / 100.0) ** 2 * 100.0 if linear >= 15.0 else 0.0
                    
                    if similarity_pct >= min_similarity:
                        metadata = entry.get('metadata', {})
                        results.append({
                            'type': normalize_type(metadata.get('type')),
                            'similarity': round(similarity_pct, 2),
                            'score': round(similarity / 100.0, 6) if not self.finder.is_similarity() else round(similarity, 6),
                            'data': {
                                'id': entry.get('id', 0),
                                'name': metadata.get('name', ''),
                                'slug': metadata.get('slug', ''),
                                'description': metadata.get('description', ''),
                                'price': metadata.get('price', 0),
                                'discount_price': metadata.get('discount_price', 0),
                                'final_price': metadata.get('discount_price', metadata.get('price', 0)),
                                'image_url': metadata.get('image_url', ''),
                                'category': metadata.get('category', ''),
                            }
                        })
                
                # Sort by similarity (descending)
                results.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Deduplicate by type+owner if rerank is enabled
                if request.rerank and len(results) > 1:
                    seen = set()
                    deduped = []
                    for r in results:
                        key = f"{r['type']}_{r['data']['id']}"
                        if key not in seen:
                            seen.add(key)
                            deduped.append(r)
                    results = deduped
                
                # Limit results
                results = results[:limit]
                
                search_time = time.time() - start_time
                
                return self._create_search_response(
                    results=results,
                    search_time=round(search_time, 3),
                    total_results=len(results),
                    error=None
                )
            
            finally:
                # Cleanup temp file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
        
        except Exception as e:
            search_time = time.time() - start_time
            return self._create_search_response(
                results=[], search_time=round(search_time, 3), total_results=0, error=str(e)
            )

    def GetStats(self, request, context):
        """Get CBIR database statistics"""
        try:
            database = load_feature_database(self.feature_db_path)
            images = database.get('images', [])
            
            total_features = 0
            for img in images:
                features = img.get('features', {})
                combined = features.get('combined') or features.get('combined_features', [])
                total_features += len(combined) if combined else 0
            
            db_size = os.path.getsize(self.feature_db_path) if os.path.exists(self.feature_db_path) else 0
            
            return self._create_stats_response(
                total_images=len(images),
                total_features=total_features,
                database_size=self._format_bytes(db_size),
                last_updated=database.get('last_updated', ''),
                status='healthy',
                message='CBIR database is operational'
            )
        
        except Exception as e:
            return self._create_stats_response(
                total_images=0, total_features=0, database_size='0 B',
                last_updated='', status='error', message=str(e)
            )

    def AddToIndex(self, request, context):
        """Add an image to the CBIR index"""
        try:
            image_path = resolve_portable_path(request.image_path)
            
            if not os.path.exists(image_path):
                return self._create_add_response(
                    success=False, message=f"File not found: {image_path}", entry_id=0
                )
            
            # Extract features
            features = self.extractor.extract(image_path)
            
            # Load database
            database = load_feature_database(self.feature_db_path)
            images = database.get('images', [])
            
            # Generate new ID
            existing_ids = [int(img.get('id') or 0) for img in images]
            new_id = (max(existing_ids) + 1) if existing_ids else 1
            
            # Create entry
            entry = {
                'id': new_id,
                'path': image_path,
                'metadata': dict(request.metadata),
                'features': {
                    'combined': np.round(features, 6).tolist(),
                    'method': 'combined',
                }
            }
            
            images.append(entry)
            database['images'] = images
            database['last_updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            
            save_feature_database(database, self.feature_db_path)
            
            return self._create_add_response(
                success=True, message="Image added to index", entry_id=new_id
            )
        
        except Exception as e:
            return self._create_add_response(
                success=False, message=str(e), entry_id=0
            )

    def RemoveFromIndex(self, request, context):
        """Remove an image from the CBIR index by metadata_id"""
        try:
            database = load_feature_database(self.feature_db_path)
            images = database.get('images', [])
            
            before_count = len(images)
            images = [
                img for img in images
                if img.get('metadata', {}).get('owner_id') != request.metadata_id
            ]
            removed = before_count - len(images)
            
            # Re-number IDs
            for i, img in enumerate(images, start=1):
                img['id'] = i
            
            database['images'] = images
            save_feature_database(database, self.feature_db_path)
            
            return self._create_remove_response(
                success=True,
                message=f"{removed} image(s) removed from index",
                removed=removed
            )
        
        except Exception as e:
            return self._create_remove_response(
                success=False, message=str(e), removed=0
            )

    def RebuildIndex(self, request, context):
        """Rebuild the entire CBIR index from dataset"""
        start_time = time.time()
        
        try:
            from src.features.build_features import build_features
            
            csv_path = request.csv_path or os.path.join(self.data_dir, 'dataset.csv')
            app_url = request.app_url or os.environ.get('LARAVEL_APP_URL', 'http://localhost:8000')
            
            if not os.path.exists(csv_path):
                return self._create_rebuild_response(
                    success=False,
                    message=f"Dataset CSV not found: {csv_path}",
                    total=0, skipped=0, elapsed_seconds=0
                )
            
            result = build_features(method='combined', csv_path=csv_path, app_url=app_url)
            
            elapsed = time.time() - start_time
            
            return self._create_rebuild_response(
                success=True,
                message=f"Rebuild complete: {result.get('total', 0)} images indexed",
                total=result.get('total', 0),
                skipped=result.get('skipped', 0),
                elapsed_seconds=round(elapsed, 2)
            )
        
        except Exception as e:
            elapsed = time.time() - start_time
            return self._create_rebuild_response(
                success=False, message=str(e),
                total=0, skipped=0, elapsed_seconds=round(elapsed, 2)
            )

    def ClearIndex(self, request, context):
        """Clear all entries from the CBIR index"""
        try:
            save_feature_database({'images': [], 'last_updated': time.strftime('%Y-%m-%dT%H:%M:%S')}, self.feature_db_path)
            
            return self._create_clear_response(
                success=True, message="Index cleared successfully"
            )
        
        except Exception as e:
            return self._create_clear_response(
                success=False, message=str(e)
            )

    def ExtractFeatures(self, request, context):
        """Extract features from an image (for debugging)"""
        try:
            if not request.image_data:
                return self._create_extract_response(
                    success=False, method='', feature_dim=0, feature_preview=[]
                )
            
            temp_file = tempfile.NamedTemporaryFile(
                dir=self.upload_folder, suffix='.jpg', delete=False
            )
            temp_file.write(request.image_data)
            temp_file.close()
            
            try:
                features = self.extractor.extract(temp_file.name)
                
                return self._create_extract_response(
                    success=True,
                    method='combined',
                    feature_dim=int(features.shape[0]),
                    feature_preview=features[:8].tolist()
                )
            finally:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
        
        except Exception as e:
            return self._create_extract_response(
                success=False, method='', feature_dim=0, feature_preview=[]
            )

    def GetHealth(self, request, context):
        """Health check endpoint"""
        return self._create_health_response(
            status='healthy',
            version='2.0.0',
            cbir_enabled=True,
            face_verify_enabled=True,
            ktp_verify_enabled=True,
            ocr_enabled=True
        )

    # Helper methods to create response objects
    #
    # Semua helper di bawah WAJIB mengembalikan objek protobuf, bukan dict.
    # Sebelumnya seluruh method ini hanya `pass`, sehingga setiap RPC
    # mengembalikan None dan gRPC menolaknya di lapisan transport dengan
    # "INTERNAL: Failed to serialize response!" -- port 50052 tetap LISTENING,
    # jadi dari luar terlihat sehat padahal semua panggilan gagal.
    def _create_search_response(self, results, search_time, total_results, error):
        response = cbir_pb2.CbirSearchResponse(
            search_time=float(search_time or 0.0),
            total_results=int(total_results or 0),
            error=str(error or ''),
        )

        for item in results or []:
            data = item.get('data') or {}
            response.results.append(
                cbir_pb2.CbirResult(
                    type=str(item.get('type') or ''),
                    similarity=float(item.get('similarity') or 0.0),
                    score=float(item.get('score') or 0.0),
                    data=cbir_pb2.CbirResultData(
                        id=int(data.get('id') or 0),
                        name=str(data.get('name') or ''),
                        slug=str(data.get('slug') or ''),
                        description=str(data.get('description') or ''),
                        price=float(data.get('price') or 0.0),
                        discount_price=float(data.get('discount_price') or 0.0),
                        final_price=float(data.get('final_price') or 0.0),
                        image_url=str(data.get('image_url') or ''),
                        category=str(data.get('category') or ''),
                    ),
                )
            )

        return response

    def _create_stats_response(self, total_images, total_features, database_size, last_updated, status, message):
        return cbir_pb2.CbirStatsResponse(
            total_images=int(total_images or 0),
            total_features=int(total_features or 0),
            database_size=str(database_size or ''),
            last_updated=str(last_updated or ''),
            status=str(status or ''),
            message=str(message or ''),
        )

    def _create_add_response(self, success, message, entry_id):
        return cbir_pb2.AddToIndexResponse(
            success=bool(success),
            message=str(message or ''),
            entry_id=int(entry_id or 0),
        )

    def _create_remove_response(self, success, message, removed):
        return cbir_pb2.RemoveFromIndexResponse(
            success=bool(success),
            message=str(message or ''),
            removed=int(removed or 0),
        )

    def _create_rebuild_response(self, success, message, total, skipped, elapsed_seconds):
        return cbir_pb2.RebuildIndexResponse(
            success=bool(success),
            message=str(message or ''),
            total=int(total or 0),
            skipped=int(skipped or 0),
            elapsed_seconds=float(elapsed_seconds or 0.0),
        )

    def _create_clear_response(self, success, message):
        return cbir_pb2.ClearIndexResponse(
            success=bool(success),
            message=str(message or ''),
        )

    def _create_extract_response(self, success, method, feature_dim, feature_preview):
        return cbir_pb2.ExtractFeaturesResponse(
            success=bool(success),
            method=str(method or ''),
            feature_dim=int(feature_dim or 0),
            feature_preview=[float(v) for v in (feature_preview or [])],
        )

    def _create_health_response(self, status, version, cbir_enabled, face_verify_enabled, ktp_verify_enabled, ocr_enabled):
        return cbir_pb2.HealthResponse(
            status=str(status or ''),
            version=str(version or ''),
            cbir_enabled=bool(cbir_enabled),
            face_verify_enabled=bool(face_verify_enabled),
            ktp_verify_enabled=bool(ktp_verify_enabled),
            ocr_enabled=bool(ocr_enabled),
        )

    def _format_bytes(self, bytes_val, precision=2):
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        for i in range(len(units) - 1):
            if bytes_val < 1024:
                return f"{round(bytes_val, precision)} {units[i]}"
            bytes_val /= 1024
        return f"{round(bytes_val, precision)} {units[-1]}"


def serve(port=50052, max_workers=10):
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # Registrasi WAJIB. Tanpa baris ini server tetap LISTENING di port tetapi
    # tidak punya handler sama sekali, sehingga setiap RPC dijawab
    # UNIMPLEMENTED. Entry point resmi (`python -m src.grpc.server`) sudah
    # mendaftarkannya; fungsi ini disamakan supaya tidak ada jalur start kedua
    # yang menghasilkan server kosong.
    cbir_pb2_grpc.add_CbirServiceServicer_to_server(CbirServiceServicer(), server)

    host = os.environ.get('AI_CORE_GRPC_HOST', '127.0.0.1')
    if host not in {'127.0.0.1', 'localhost', '::1'} and os.environ.get('AI_CORE_GRPC_ALLOW_INSECURE_PUBLIC') != 'true':
        raise RuntimeError('Refusing public insecure gRPC bind; use TLS or explicit private-network opt-in.')
    server.add_insecure_port(f'{host}:{port}')
    server.start()
    
    print(f"AI Core gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("AI Core gRPC server stopped")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Core gRPC Server')
    parser.add_argument('--port', type=int, default=50052, help='gRPC server port')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads')
    
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.workers)
