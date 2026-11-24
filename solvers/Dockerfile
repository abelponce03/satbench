# syntax=docker/dockerfile:1.7
# Reproducible build of every solver of manifest.toml.
#
#   uv run python -m satbench.solvers.build fetch --manifest solvers/manifest.toml --cache solvers/archives
#   docker build -f solvers/Dockerfile -t satbench-solvers:2026.11 .
#
# The base image is pinned by digest. The exact version of every installed package is recorded
# in /opt/toolchain-packages.txt, and the built image itself is archived with the study (Docker
# archive / Apptainer SIF), so the binaries can be reused rather than rebuilt. Solver builds run
# without network access.

ARG BASE=ubuntu:24.04@sha256:9c7f200c07e3d4a03e7249d2c16bf66a51120fb876aa2242cf31451412543289
FROM ${BASE} AS toolchain
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential clang lld cmake pkg-config zlib1g-dev libgmp-dev \
      python3 xz-utils bash ca-certificates \
 && dpkg-query -W -f '${Package}\t${Version}\n' > /opt/toolchain-packages.txt \
 && rm -rf /var/lib/apt/lists/*

FROM toolchain AS build
COPY solvers/manifest.toml /opt/satbench/manifest.toml
COPY src/satbench/solvers/build.py /opt/satbench/build.py
COPY solvers/archives /opt/archives
RUN --network=none python3 /opt/satbench/build.py build \
      --manifest /opt/satbench/manifest.toml --cache /opt/archives --out /opt/solvers --jobs 8 \
  ; cat /opt/solvers/build-report.json > /dev/null

# satbench itself, installed from uv.lock (exact dependency versions). Solvers and experiments run
# inside this image, never directly on the host: the solvers are third-party code.
FROM toolchain AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/satbench/.venv UV_PYTHON_DOWNLOADS=never UV_PYTHON=/usr/bin/python3
COPY pyproject.toml uv.lock README.md /opt/satbench/
COPY src /opt/satbench/src
RUN cd /opt/satbench && uv sync --frozen --no-dev
ENV PATH=/opt/satbench/.venv/bin:$PATH

FROM runtime AS solvers
COPY --from=build /opt/solvers /opt/solvers
COPY solvers/manifest.toml /opt/solvers/manifest.toml
