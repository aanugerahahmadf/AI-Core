# -*- coding: utf-8 -*-
"""Backward-compatible entry point (shim)."""

from src.app import *  # noqa: F401,F403

if __name__ == "__main__":
    from src.run_all import main
    main()