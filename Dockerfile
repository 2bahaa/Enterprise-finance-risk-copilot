FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch first. The default Linux wheel pulls in gigabytes of
# CUDA libraries that this project cannot use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src

# Run as a normal user. /models holds the embedding model cache and is
# created here so a named volume mounted on it inherits this ownership.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models \
    && chown -R app:app /models /app
USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]