# Use Python as the base image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first
COPY requirements.txt .

# Install dependencies and security-fixed packages in one image layer
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel msgpack urllib3 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --upgrade setuptools msgpack urllib3 \
    && python -m pip check

# Copy application source code
COPY src/ ./src/

# Copy trained model artifacts
COPY models/ ./models/

# Expose the FastAPI port
EXPOSE 8080

# Start the FastAPI application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
