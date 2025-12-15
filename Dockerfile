FROM python:3.11-slim AS core

# Install system dependencies (git, build-essential, etc.)
# We add --no-install-recommends and clean up apt lists to keep the image slim.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    pkg-config \
    libhdf5-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use the system-wide Python (no venv needed in container)
ENV UV_SYSTEM_PYTHON=1
WORKDIR /home/app/mava

# Build-time argument to control GPU installation.
# Build with: docker build --build-arg USE_CUDA=true -t my-image .
ARG USE_CUDA=false

# --- Dependency Installation Layer ---
# Copy only the file needed for dependency resolution
COPY pyproject.toml .
COPY uv.lock .

# Install all dependencies *except* the local project
# We use a shell variable (JAX_EXTRA) to conditionally add the 'cuda12' extra
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_CUDA" = true ] ; then \
        uv sync --locked --no-install-project --extra cuda12  ; \
    else \
        uv sync --locked --no-install-project ; \
    fi

# --- Application Code Layer ---
# This layer is cached and only re-runs if Mava code changes.

# Copy all the application source code
COPY . .

# Install the local project itself
# We pass the JAX_EXTRA variable again to ensure the
# full dependency set (including the project) is synced correctly.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_CUDA" = true ] ; then \
        uv sync --locked --extra cuda12  ; \
    else \
        uv sync --locked ; \
    fi

# Expose Tensorboard port
EXPOSE 6006
