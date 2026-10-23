# -*- coding: utf-8 -*-
"""Service layer (placeholder) untuk AI Core."""

from __future__ import annotations


def service_status() -> dict:
    return {"status": "ok", "service": "ai-core"}


__all__ = ["service_status"]