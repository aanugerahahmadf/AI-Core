# Wedding CBIR AI Core (Python Flask) — CPU build without Laragon
# Based on python:3.12 slim. PyTorch CPU wheel to keep image smaller.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# --- System deps needed by OpenCV, easyocr, torchvision, etc. ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install torch (CPU) first, then the rest
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

# Copy the application source
COPY . .

# Data / uploads dirs
RUN mkdir -p data uploads reports

EXPOSE 5000

# Run Flask. Reads FLASK_HOST / FLASK_PORT from .env via app.py
CMD ["python", "app.py"]
