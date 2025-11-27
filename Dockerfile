# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY api.py .
COPY config.yaml .

# Copy ML analysis module
COPY ML_analysis/ ./ML_analysis/

# Copy thermal analysis module
COPY heat_point_analysis/ ./heat_point_analysis/

# Copy the trained model
COPY ML_analysis/models/best_model.pth ./ML_analysis/models/best_model.pth

# Create directory for temporary files
RUN mkdir -p /tmp/thermal_uploads

# Expose port (Render will override this with PORT env variable)
EXPOSE 10000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Run the application
# Render sets PORT env variable, so we use it
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}
