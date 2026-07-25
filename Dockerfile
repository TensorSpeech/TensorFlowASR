FROM tensorflow/tensorflow:2.20.0-gpu

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get -y install apt-utils gcc libpq-dev libsndfile-dev git build-essential cmake screen

# Clear cache
RUN apt clean && apt-get clean

WORKDIR /app

# Install into the image's existing interpreter rather than a nested .venv
ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first so they stay cached across source changes
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra cuda --no-install-project

# Install the project itself
COPY tensorflow_asr ./tensorflow_asr
RUN uv sync --frozen --extra cuda

# Install rnnt_loss
COPY scripts /scripts
ARG install_rnnt_loss=true
ARG using_gpu=true
RUN if [ "$install_rnnt_loss" = "true" ] ; \
    then if [ "$using_gpu" = "true" ] ; then export CUDA_HOME=/usr/local/cuda ; else echo 'Using CPU' ; fi \
    && ./scripts/install_rnnt_loss.sh \
    else echo 'Using pure TensorFlow'; fi

RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /root/.bashrc
