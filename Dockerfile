FROM python:3.12-slim

# System deps required by opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps — strip the "python==x.x.x" line that pip cannot handle
COPY Requirement.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^python==" Requirement.txt \
    | sed 's/opencv-python==.*/opencv-python-headless==4.13.0.90/' \
    > requirements_clean.txt \
    && pip install --timeout 1000 --retries 5 -r requirements_clean.txt \
    && rm requirements_clean.txt

# Copy source (models are mounted as volumes at runtime)
COPY app.py config.py seed.py ./
COPY core/ ./core/
COPY routes/ ./routes/
COPY schemas/ ./schemas/
COPY services/ ./services/
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

ENTRYPOINT ["./entrypoint.sh"]
