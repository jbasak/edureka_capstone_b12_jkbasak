# ── Procurement Agent — Dockerfile ───────────────────────────────────────────
# Base: slim Python 3.11 for small image footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies needed by PyMuPDF and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first to maximise Docker layer cache reuse
COPY requirements.txt .

# Install Python dependencies
# --extra-index-url for CPU-only torch is declared inside requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
