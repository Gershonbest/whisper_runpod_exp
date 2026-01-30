# Stage 1: Build Stage
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 and system dependencies
# Ubuntu 22.04 ships with Python 3.10, but we need 3.11, so we'll use deadsnakes PPA
RUN apt-get -y update --fix-missing && \
    apt-get -y install --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get -y update && \
    apt-get -y install --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt/program

COPY ./pyproject.toml /opt/program/
# Sync dependencies (uv will create uv.lock if it doesn't exist)
# Using --no-install-project since this is an application, not a library
RUN uv sync --no-install-project 

# Stage 2: Runtime Stage
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PATH="/opt/program:${PATH}"
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /opt/program

# Install Python 3.11 and system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    ffmpeg \
    wget \
    cmake \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Copy Python packages from build stage
# uv installs to .venv by default, copy the entire venv
COPY --from=build /opt/program/.venv /opt/program/.venv

# Activate the virtual environment
ENV PATH="/opt/program/.venv/bin:${PATH}"

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

RUN ldconfig

EXPOSE 8000

COPY ./build_code/src/ /opt/program/

ENTRYPOINT ["python3", "main.py"]

