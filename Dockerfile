FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models results

# Set environment variables
ENV PYTHONPATH=/app
ENV NVIDIA_API_KEY=

# Expose API port
EXPOSE 8000

# Run the complete pipeline and start API server
CMD ["sh", "-c", "python create_demo_data.py && python preprocessing.py && python embed_geneformer_enhanced.py && python train_classifier.py && python evaluate.py && python serve_api.py"]
