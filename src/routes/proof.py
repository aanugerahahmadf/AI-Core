# -*- coding: utf-8 -*-
"""POST /api/proof/verify - validasi bukti pembayaran via OCR (FaceService::verifyProof())."""

from __future__ import annotations

from .context import *  # noqa: F401,F403 - runtime bersama
from .context import (
    _blur_score,
    _ocr_text,
    _extract_amounts,
    _to_float,
)  # helper privat: tidak ikut star-import


# ---------------------------------------------------------------------------
# Computer Vision — Verifikasi Bukti Pembayaran (OCR nominal)
# ---------------------------------------------------------------------------
def register(app):
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
