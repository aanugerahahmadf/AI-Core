"""
AI Core gRPC Package
Wedding Flower Decoration App

This package contains the gRPC server implementation for the AI Core service.
It provides CBIR (Content-Based Image Retrieval) capabilities for the
Wedding Flower Decoration App.
"""

import site, importlib.util, os, sys

# Load the installed grpcio module under the name 'grpc' before any submodule
# does `import grpc`, otherwise the local `src/grpc` package would shadow it.
_grpcio = None
for _base in site.getsitepackages() + [site.getusersitepackages()]:
    _candidate = os.path.join(_base, "grpc", "__init__.py")
    if os.path.exists(_candidate):
        _spec = importlib.util.spec_from_file_location("grpc", _candidate)
        _grpcio = importlib.util.module_from_spec(_spec)
        sys.modules["grpc"] = _grpcio
        _spec.loader.exec_module(_grpcio)
        break

__version__ = '2.0.0'
__author__ = 'Wedding Flower Decoration App Team'