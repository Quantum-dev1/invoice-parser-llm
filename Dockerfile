# syntax=docker/dockerfile:1
# HTTP API image. Configure the LLM at runtime with INVOICE_LLM_* environment variables.

FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
COPY --from=build /dist/*.whl /tmp/
RUN WHEEL="$(ls /tmp/*.whl)" && pip install "${WHEEL}[api]" && rm -f /tmp/*.whl \
    && useradd --create-home --uid 10001 app
USER app
WORKDIR /home/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["uvicorn", "invoice_parser_llm.api:app", "--host", "0.0.0.0", "--port", "8000"]
