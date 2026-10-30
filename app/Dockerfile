FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/udeshikarmu/tea-yield-prediction"

WORKDIR /app

# Install system dependencies (including libgomp1 required by CatBoost/SHAP on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application, dataset, and source code
COPY config/ config/
COPY data/ data/
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY app/ app/
RUN mkdir -p logs

# Ensure model & SHAP artifacts are compatible with the container's Linux Python 3.11 runtime
RUN python src/preprocess.py && python src/train.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
