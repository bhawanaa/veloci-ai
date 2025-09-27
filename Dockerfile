# --- Build stage (tiny, no compilers needed here but keeps pattern ready) ---
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install only needed OS packages and keep image small
# (ca-certificates for TLS; clean apt cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils libgl1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# If you need to trust a corporate root CA, uncomment these lines and
# copy your PEM file next to the Dockerfile:
COPY corp-ca.pem /usr/local/share/ca-certificates/corp-ca.crt
RUN update-ca-certificates

# Make sure Python libs (httpx/requests) see the system bundle
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV PYTHONHTTPSVERIFY=0

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .
COPY static/ static/
COPY templates/ templates/
COPY test_openai_connectivity.py .
COPY patched_edge_tts/ patched_edge_tts/
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

