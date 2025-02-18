# Stage 1: Build environment
FROM python:3.12-slim AS core

# Add git
RUN apt-get update && apt-get install -y git build-essential pkg-config libhdf5-dev

# Add uv and use the system python (no need to make venv)
USER root
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /home/app/mava

COPY . .

RUN uv pip install -e .

ARG USE_CUDA=false
RUN if [ "$USE_CUDA" = true ] ; \
    then uv pip install jax[cuda12]==0.4.30 ; \
    fi

EXPOSE 6006
