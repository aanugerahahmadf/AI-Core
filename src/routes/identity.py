# -*- coding: utf-8 -*-
"""POST /api/ktp/verify - validasi dokumen identitas KTP/SIM/NPWP/passport (FaceService::verifyKtp())."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .context import (
    _detect_faces,
    _blur_score,
    _ocr_text,
    _detect_identity_document,
    _aspect_score,
    _id_doc_max_score,
    _best_identity_doc_type,
)  # helper privat: tidak ikut star-import


def register(app):
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
