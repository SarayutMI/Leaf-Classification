# ── base ─────────────────────────────────────────────────────────
# Shared by the test and runtime targets so CI tests the same
# interpreter, deps and source layout that production runs.
FROM python:3.12-slim AS base

# System deps required by opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch is not in Requirement.txt — ultralytics pulls it in, and PyPI's
# default wheel is the CUDA build: torch (527 MB) plus nvidia-cudnn-cu13
# (366 MB), nvidia-cusparselt-cu13 (170 MB) and cuda-toolkit, none of which
# a CPU-only host can use. Install the CPU wheels first so the resolver
# below sees the requirement already satisfied.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 1000 --retries 5 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# Install Python deps — strip the "python==x.x.x" line that pip cannot handle.
#
# Deliberately the full opencv-python, not the headless build: ultralytics
# reads `cv2.imshow` at import time (utils/patches.py), which headless does
# not define, so the API dies during model load. libgl1 and libglib2.0-0
# above are what the full build needs.
COPY Requirement.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^python==" Requirement.txt > requirements_clean.txt \
    && pip install --timeout 1000 --retries 5 -r requirements_clean.txt \
    && rm requirements_clean.txt

# triton ships with the CUDA torch wheel and segfaults on import here. The
# CPU wheel should not pull it, but stay defensive — this is cheap.
RUN pip uninstall -y triton || true

# Copy source (models are mounted as volumes at runtime)
COPY src/ ./src/
COPY scripts/ ./scripts/

# ── test ─────────────────────────────────────────────────────────
# Built by the Jenkins Test stage. Source is COPYed in, never
# bind-mounted, so the run cannot leave root-owned files in the
# workspace for cleanWs() to choke on.
FROM base AS test

COPY tests/ ./tests/
CMD ["pytest", "-q", "tests"]

# ── runtime ──────────────────────────────────────────────────────
# Last stage, so a plain `docker build` / `docker compose build`
# produces the runtime image.
FROM base AS runtime

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

ENTRYPOINT ["./entrypoint.sh"]
