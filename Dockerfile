FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY dataset.py .
COPY models.py .
COPY loss.py .
COPY train.py .
COPY resume_training.py .
COPY predict.py .

# Default: run inference
CMD ["python", "predict.py", "--help"]
