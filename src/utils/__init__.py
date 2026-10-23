# -*- coding: utf-8 -*-
"""Utility helpers for AI Core."""

from __future__ import annotations

import os


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def allowed_image(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def allowed_video(filename: str) -> bool:
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))


__all__ = ["ensure_dir", "allowed_image", "allowed_video"]