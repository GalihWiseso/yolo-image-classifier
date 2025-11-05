# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Reduce memory/threads usage for numerical libs
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	NUMEXPR_MAX_THREADS=1

# No additional system libs required for regression stack

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000

# Lean server: single worker, short keep-alive
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header", "--timeout-keep-alive", "5"]
