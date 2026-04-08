# ============================================================
# Dockerfile — Smart Monsoon Farm (FINAL FIXED VERSION)
# ============================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY deployment/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install fastapi uvicorn

# Copy project files
COPY env/       ./env/
COPY tasks/     ./tasks/
COPY grader/    ./grader/
COPY baseline/  ./baseline/
COPY configs/   ./configs/
COPY app/       ./app/

# Copy api file (IMPORTANT)
COPY api.py ./api.py

# Ensure init files exist
RUN touch env/__init__.py tasks/__init__.py grader/__init__.py baseline/__init__.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SEED=42

# ============================================================
# RUN API SERVER (REQUIRED FOR GRADER)
# ============================================================

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]