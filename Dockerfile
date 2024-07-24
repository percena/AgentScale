# Use Python 3.11 as the base image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* /app/

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy project
COPY . /app

# Set the Python path to include the src directory
ENV PYTHONPATH=/app/src

# Expose port 9000 for the RAG API service
EXPOSE 9000

# Use environment variable to determine which service to run
CMD ["sh", "-c", "if [ \"$SERVICE\" = \"rag_api\" ]; then python src/agentscale/rag/app.py; else python src/agentscale/main.py; fi"]