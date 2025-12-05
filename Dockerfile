# Use Python 3.11 slim as base image (compatible with mediapipe)
FROM python:3.11-slim

# Install system libraries needed by mediapipe & OpenCV backend (libGL, libglib)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory inside the container
WORKDIR /app

# Copy dependency file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Railway exposes PORT dynamically; fallback to 8080 if missing
ENV PORT=8080

# Start the Flask application via gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
