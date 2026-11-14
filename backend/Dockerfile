# Build from the repo root:  docker build -f backend/Dockerfile -t enzera-backend .
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY ml/ ml/
COPY backend/ backend/
COPY models/ models/
COPY data/demo/ data/demo/
COPY reports/metrics.json reports/metrics.json
ENV DB_PATH=/tmp/enzera.db
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
