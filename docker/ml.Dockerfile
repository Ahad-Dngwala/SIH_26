# Layer 1 (data/models) + the Python half of Layer 2 (fusion_core/python_prototype,
# map_matching, tools/benchmark_replay) all run in this one image. See
# docker-compose.yml - the repo is bind-mounted in at runtime, this image
# only bakes in the interpreter and dependencies, so editing code on the
# host does not require a rebuild.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Repo root is bind-mounted onto /workspace by docker-compose, so
# `from models.common.normalization import ...` etc. resolve without an
# editable install.
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["bash"]
