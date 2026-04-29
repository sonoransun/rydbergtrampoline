# Reproducible image for Rydberg Trampoline.
#
# Pre-installs the heavy backends (QuSpin, TeNPy) so contributors can avoid
# the Cython/C++ build dance on their host. Build once, mount the repo:
#
#   docker build -t rydberg-trampoline:dev .
#   docker run --rm -it -v "$PWD":/work -w /work rydberg-trampoline:dev pytest -q

FROM python:3.12-slim

# Build deps for QuSpin / TeNPy (Cython + C++ + BLAS).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /work
COPY pyproject.toml ./
COPY README.md ./
COPY rydberg_trampoline ./rydberg_trampoline

# Install everything; the image is meant for development, not for slim deploys.
RUN python -m pip install --upgrade pip \
    && pip install -e .[all,dev]

CMD ["pytest", "-q"]
