"""
Wedding Organizer CBIR AI Core Package
This package contains the core AI modules for Content-Based Image Retrieval (CBIR)
for the Wedding Organizer application.
"""

__version__ = "0.1.0"
__author__ = "Weeding Organizer Team"

# Import main modules for easier access
from . import addons
from . import dataset
from . import features
from . import models

__all__ = [
    "addons",
    "dataset", 
    "features",
    "models",
]
