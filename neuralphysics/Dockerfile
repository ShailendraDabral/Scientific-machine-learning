# syntax=docker/dockerfile:1
# Stage 1: build the front end.
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python back end that also serves the built front end.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# CPU-only PyTorch keeps the image small; swap the index URL for a CUDA build if needed.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY core/ core/
COPY api/ api/
RUN pip install --no-deps -e core -e api
COPY --from=frontend /build/dist frontend/dist
RUN useradd --create-home neuralphysics && chown -R neuralphysics /app
USER neuralphysics
WORKDIR /app/api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
