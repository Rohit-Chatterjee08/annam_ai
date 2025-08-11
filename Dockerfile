# Annam AI - Agricultural Intelligence System
# Multi-stage Docker build for production deployment

# Build stage
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Create data directories
RUN mkdir -p data/raw data/processed data/models logs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (FastAPI)
CMD ["uvicorn", "deployment.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands for different deployments:
# Streamlit: streamlit run deployment/streamlit/app.py --server.port 8501 --server.address 0.0.0.0
# Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
