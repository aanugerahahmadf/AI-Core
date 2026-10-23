# -*- coding: utf-8 -*-
"""
routes/ — Package registrasi seluruh route AI Core.

Satu endpoint = satu file. Semuanya saling terhubung lewat dua jangkar:
  * routes/context.py : runtime bersama (config, helper, singleton); semua
                        file route membuka `from .context import *`
  * register_all(app) : dipanggil src/app.py setelah Flask app dibuat;
                        memasang register(app) dari tiap file route.

Urutan registrasi di bawah identik dengan urutan definisi di app.py monolit
yang lama.
"""
from __future__ import annotations

from . import (
    home,
    health,
    status,
    index_stats,
    search,
    index_add,
    index_remove,
    index_clear,
    index_rebuild,
    sync,
    features,
    evaluate,
    face,
    identity,
    proof,
    errors,
    arithmetic,
    arithmetic_ops,
)


_ROUTES = (
    ('home', 'GET  /'),
    ('health', 'GET  /health'),
    ('status', 'GET  /status'),
    ('index_stats', 'GET  /api/index/stats'),
    ('search', 'POST /api/search'),
    ('index_add', 'POST /api/index/add'),
    ('index_remove', 'POST /api/index/remove'),
    ('index_clear', 'POST /api/index/clear'),
    ('index_rebuild', 'POST /api/index/rebuild-from-dataset'),
    ('sync', 'POST /api/sync'),
    ('features', 'POST /api/features/extract'),
    ('evaluate', 'GET  /api/evaluate'),
    ('face', 'POST /api/face/verify'),
    ('identity', 'POST /api/ktp/verify'),
    ('proof', 'POST /api/proof/verify'),
    ('errors', 'Handler 413/404/500'),
    ('arithmetic', 'POST /api/arithmetic'),
    ('arithmetic_ops', 'GET  /api/arithmetic/ops'),
)


def register_all(app):
    """Pasang seluruh route package ini ke aplikasi Flask `app`."""
    for _name, _desc in _ROUTES:
        globals()[_name].register(app)
