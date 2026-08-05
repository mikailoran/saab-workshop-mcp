# Match the local dev environment's Python version for behavioral parity.
FROM python:3.14-slim

# torch's CPU backend dynamically links libgomp at runtime; the slim base
# doesn't ship it, so sentence-transformers import fails without this.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
WORKDIR /app

# Cache dir for the embedding model, shared between the build-time download
# below and the runtime load in retrieval.py -- keeping it under /app means
# the single chown further down covers code, index, and cache alike.
ENV HF_HOME=/app/.cache

COPY requirements.txt .
# --index-url points pip at PyTorch's CPU-only wheels (the default index
# resolves torch to a CUDA build with several GB of unneeded GPU libraries,
# since this service only ever does CPU inference); --extra-index-url falls
# back to PyPI for the rest of requirements.txt, which isn't hosted there.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# Bake the embedding model weights into the image at build time, so the
# container serves queries fully offline and the first request isn't slowed
# down by (or dependent on) a HuggingFace Hub download.
COPY retrieval.py .
RUN python -c "from retrieval import EMBEDDING_MODEL; from sentence_transformers import SentenceTransformer; SentenceTransformer(EMBEDDING_MODEL)"

# The Chroma index is a prebuilt, read-only artifact produced offline by
# ingest.py -- ship it in the image rather than mounting a volume.
COPY index/ ./index/
COPY mcp_server.py .

RUN chown -R appuser:appuser /app
USER appuser

ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000
EXPOSE 8000

CMD ["python", "mcp_server.py"]
