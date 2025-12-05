# Use Python 3.11 (compatible with mediapipe)
FROM python:3.11-slim

# Install system libraries needed by mediapipe / cv2 (libGL etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Railway exposes $PORT; default to 8080 just in case
ENV PORT=8080

# Start the app with gunicorn (app.py -> app variable)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
