FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for binary wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for caching
COPY requirements-render.txt .

# CRITICAL: Force ONLY binary wheels.
RUN pip install --no-cache-dir --only-binary=:all: -r requirements-render.txt

# Copy the rest of the application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
