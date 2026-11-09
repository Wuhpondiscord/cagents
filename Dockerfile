FROM node:22-bookworm-slim AS browser
WORKDIR /build
RUN npm install --no-audit --no-fund --ignore-scripts @wllama/wllama@3.8.1 \
    && test -f node_modules/@wllama/wllama/esm/wasm/wllama.wasm

FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt
COPY app.py agent.py ./
COPY web/ ./web/
COPY --from=browser /build/node_modules/@wllama/wllama/esm/ ./web/vendor/
USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log", "--ws-max-size", "4194304"]
