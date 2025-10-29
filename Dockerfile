# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

# System deps for OpenCV/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libgl1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . .

# (Optional) Warm up YOLO weights during build for faster first request
# This requires network in build environment
RUN python3 - <<'PY'
from ultralytics import YOLO
try:
	model = YOLO('yolov8n-cls.pt')
	# Use a tiny image to force a one-time weights download and compile
	model.predict(source='https://ultralytics.com/images/bus.jpg', imgsz=224, device='cpu', save=False, verbose=False)
except Exception as e:
	print('Warmup skipped:', e)
PY

EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
