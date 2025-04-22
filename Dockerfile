FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with better error handling
RUN pip install --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Railway automatically sets PORT environment variable
ENV PORT=8000

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app