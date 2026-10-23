# -*- coding: utf-8 -*-
"""Entry point GraphQL (placeholder) untuk AI Core."""

from __future__ import annotations

import os
from flask import Flask, jsonify

from .resolvers import health_resolver

app = Flask(__name__)


@app.route("/graphql", methods=["GET", "POST"])
def graphql_endpoint():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.graphql")
    schema = ""
    if os.path.exists(schema_path):
        with open(schema_path, encoding="utf-8") as f:
            schema = f.read()
    return jsonify({"data": {"health": health_resolver()}, "schema": schema})


def main():
    print("AI Core GraphQL server (placeholder) running on port 5001")
    app.run(host="0.0.0.0", port=5001, debug=False)


if __name__ == "__main__":
    main()