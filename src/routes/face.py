# -*- coding: utf-8 -*-
"""POST /api/face/verify - verifikasi wajah selfie vs referensi (FaceService::verifyFace())."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .context import (
    _detect_faces_mtcnn,
    _detect_faces,
    _blur_score,
    _eyes_open,
    _save_crop,
    _face_similarity,
)  # helper privat: tidak ikut star-import


def register(app):
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
