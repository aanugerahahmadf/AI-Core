# -*- coding: utf-8 -*-
"""Jalankan REST + gRPC + GraphQL dalam satu proses (satu server)."""

from __future__ import annotations

import os
import threading
import time

# REST
from src.routes.server import app as rest_app
# Utils & services (ensure connected)
from src import utils, services  # noqa: F401

# gRPC
import src.grpc.server as grpc_server

# GraphQL (placeholder)
from src.graphql.server import app as gql_app


def run_rest():
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    rest_app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


def run_graphql():
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    gql_app.run(host=host, port=5001, debug=False, use_reloader=False)


def main():
    t_rest = threading.Thread(target=run_rest, daemon=True)
    t_gql = threading.Thread(target=run_graphql, daemon=True)
    t_rest.start()
    t_gql.start()
    time.sleep(1)
    print("REST/gRPC/GraphQL started on configured private interfaces")
    # gRPC blocks (main thread, signal handling ok)
    grpc_server.serve(port=50052, max_workers=10)


if __name__ == "__main__":
    main()
