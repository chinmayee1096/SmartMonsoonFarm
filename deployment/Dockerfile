# ============================================================
# Dockerfile — Smart Monsoon-Resilient Hydroponic Farm
# Lightweight multi-stage build
# ============================================================

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY deployment/requirements.txt ./requirements.txt

# Install Python dependencies
# Use CPU-only torch to keep image size manageable (~2GB vs 6GB)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy source code
# ============================================================
COPY env/           ./env/
COPY tasks/         ./tasks/
COPY grader/        ./grader/
COPY baseline/      ./baseline/
COPY configs/       ./configs/
COPY app/           ./app/

# Create empty __init__ files if missing
RUN touch env/__init__.py tasks/__init__.py grader/__init__.py baseline/__init__.py

# ============================================================
# Environment variables
# ============================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SEED=42

# ============================================================
# Default: run heuristic inference on medium task
# Override with: docker run ... python baseline/inference.py --task hard --agent ppo
# ============================================================
CMD ["python", "baseline/inference.py", "--task", "medium", "--agent", "heuristic", "--verbose"]

# ============================================================
# Streamlit target (for HF Spaces)
# Build: docker build --target streamlit -t monsoon-farm-ui .
# ============================================================
FROM base AS streamlit

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
