FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for binary wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version which handles manylinux wheels better
RUN pip install --upgrade pip

# Copy requirements first for caching
COPY requirements-render.txt .

# CRITICAL FLAG: Force pip to ONLY download pre-compiled binary wheels.
# This prevents any Rust/C compilation from happening.
RUN pip install --no-cache-dir --only-binary=:all: -r requirements-render.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
